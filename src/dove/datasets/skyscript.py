"""
File: src/dove/datasets/skyscript.py
Purpose:
    SkyScript retrieval dataset and utilities.

    - Load image-text pairs from a CSV (filepath + caption field).
    - Best-effort meta loading for each image:
        * SkyScript meta['box'] is a GEO bounding box (lon/lat) => MUST NOT be used
          as a pixel bbox.
        * Optionally accept a dedicated pixel/normalized bbox stored under keys like
          'center_box', 'pixel_bbox', 'roi', etc. (policy='meta_pixel').
    - Provide torchvision transforms suitable for remote-sensing images.
    - Provide a collate function that returns:
        images: (B, 3, H, W)
        input_ids: (B, T)
        lengths: (B,)
        attn_mask: (B, T)
        center_boxes: (B, 4) normalized xyxy or -1 if unavailable.
"""

from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as tvf


ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------------------------------------------------------------
# Constants / Regex
# -----------------------------------------------------------------------------
_IMAGES_SEG_RE = re.compile(r"^images(\d+)$", flags=re.IGNORECASE)
_DEFAULT_META_EXTS: Tuple[str, ...] = (".pickle", ".pkl", ".json")


# -----------------------------------------------------------------------------
# Transform helpers (torchvision compatibility)
# -----------------------------------------------------------------------------
def _tv_resize(
    size: Any,
    *,
    interpolation: transforms.InterpolationMode,
    antialias: bool = True,
) -> transforms.Resize:
    """Create Resize with backward-compatible 'antialias' handling."""
    try:
        return transforms.Resize(size, interpolation=interpolation, antialias=antialias)
    except TypeError:
        return transforms.Resize(size, interpolation=interpolation)


def _tv_random_resized_crop(
    size: int,
    *,
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
    interpolation: transforms.InterpolationMode,
    antialias: bool = True,
) -> transforms.RandomResizedCrop:
    """Create RandomResizedCrop with backward-compatible 'antialias' handling."""
    try:
        return transforms.RandomResizedCrop(
            size,
            scale=scale,
            ratio=ratio,
            interpolation=interpolation,
            antialias=antialias,
        )
    except TypeError:
        return transforms.RandomResizedCrop(
            size,
            scale=scale,
            ratio=ratio,
            interpolation=interpolation,
        )


class RandomRotate90(torch.nn.Module):
    """Rotate image by {0, 90, 180, 270} degrees (remote-sensing friendly)."""

    def forward(self, img: Any) -> Any:  # noqa: ANN401
        k = int(torch.randint(0, 4, (1,)).item())
        angle = 90 * k
        if angle == 0:
            return img
        return tvf.rotate(img, angle)


def build_skyscript_transform(
    image_size: int = 256,
    *,
    train: bool = False,
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
    imagenet_norm: bool = True,
    random_rotation: bool = False,
    color_jitter: float = 0.0,
) -> transforms.Compose:
    """
    Build torchvision transform for SkyScript-like remote-sensing retrieval.

    Args:
        image_size: Output image size (H=W=image_size).
        train: Whether to use training augmentations.
        interpolation: Interpolation mode for resize/crop.
        imagenet_norm: Apply ImageNet mean/std normalization.
        random_rotation: Random rotate 0/90/180/270.
        color_jitter: Strength of color jitter (0 disables).

    Returns:
        torchvision.transforms.Compose
    """
    tfms: List[Any] = []

    if train:
        tfms.append(_tv_resize(int(image_size * 1.12), interpolation=interpolation))
        tfms.append(
            _tv_random_resized_crop(
                image_size,
                scale=(0.67, 1.00),
                ratio=(0.50, 2.00),
                interpolation=interpolation,
            )
        )

        if random_rotation:
            tfms.append(RandomRotate90())

        cj = float(color_jitter)
        if cj > 0.0:
            tfms.append(
                transforms.ColorJitter(
                    brightness=cj,
                    contrast=cj,
                    saturation=cj,
                    hue=min(0.1, cj / 4.0),
                )
            )

        tfms.append(transforms.RandomHorizontalFlip(p=0.5))
    else:
        tfms.append(_tv_resize((image_size, image_size), interpolation=interpolation))

    tfms.append(transforms.ToTensor())

    if imagenet_norm:
        tfms.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

    return transforms.Compose(tfms)


# -----------------------------------------------------------------------------
# Center-box utilities
# NOTE: SkyScript meta['box'] is lon/lat => MUST NOT be treated as pixel bbox.
# -----------------------------------------------------------------------------
def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _sort_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> Tuple[float, float, float, float]:
    xa, xb = (x1, x2) if x1 <= x2 else (x2, x1)
    ya, yb = (y1, y2) if y1 <= y2 else (y2, y1)
    return xa, ya, xb, yb


def _looks_normalized_xyxy(xyxy: Tuple[float, float, float, float]) -> bool:
    """Heuristic: values already in [0, 1] (allow small margin)."""
    x1, y1, x2, y2 = xyxy
    return (
        0.0 <= x1 <= 1.5
        and 0.0 <= y1 <= 1.5
        and 0.0 <= x2 <= 1.5
        and 0.0 <= y2 <= 1.5
    )


def _looks_like_latlon_box(xyxy: Tuple[float, float, float, float]) -> bool:
    """
    SkyScript meta['box'] is (west_lon, south_lat, east_lon, north_lat).
    Reject if it fits lon/lat ranges.
    """
    x1, y1, x2, y2 = xyxy
    return (
        -180.0 <= x1 <= 180.0
        and -180.0 <= x2 <= 180.0
        and -90.0 <= y1 <= 90.0
        and -90.0 <= y2 <= 90.0
    )


def _try_parse_xyxy(box: Any) -> Optional[Tuple[float, float, float, float]]:  # noqa: ANN401
    """Parse xyxy from list/tuple or dict with common key patterns."""
    if box is None:
        return None

    if isinstance(box, dict):
        key_map = {str(k).lower(): k for k in box.keys()}
        patterns = [
            ("x1", "y1", "x2", "y2"),
            ("xmin", "ymin", "xmax", "ymax"),
            ("left", "top", "right", "bottom"),
        ]
        for a, b, c, d in patterns:
            if a in key_map and b in key_map and c in key_map and d in key_map:
                try:
                    return (
                        float(box[key_map[a]]),
                        float(box[key_map[b]]),
                        float(box[key_map[c]]),
                        float(box[key_map[d]]),
                    )
                except (TypeError, ValueError):
                    return None
        return None

    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            return tuple(float(v) for v in box)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None

    return None


def _try_parse_xywh(box: Any) -> Optional[Tuple[float, float, float, float]]:  # noqa: ANN401
    """Parse xywh from list/tuple or dict with common key patterns."""
    if box is None:
        return None

    if isinstance(box, dict):
        key_map = {str(k).lower(): k for k in box.keys()}
        patterns = [
            ("x", "y", "w", "h"),
            ("xmin", "ymin", "width", "height"),
        ]
        for a, b, c, d in patterns:
            if a in key_map and b in key_map and c in key_map and d in key_map:
                try:
                    return (
                        float(box[key_map[a]]),
                        float(box[key_map[b]]),
                        float(box[key_map[c]]),
                        float(box[key_map[d]]),
                    )
                except (TypeError, ValueError):
                    return None
        return None

    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            return tuple(float(v) for v in box)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None

    return None


def _normalize_xyxy(
    xyxy: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    """Convert pixel xyxy to normalized [0,1] xyxy if needed."""
    x1, y1, x2, y2 = _sort_xyxy(*xyxy)

    if _looks_normalized_xyxy((x1, y1, x2, y2)):
        return _sort_xyxy(_clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2))

    if img_w <= 0 or img_h <= 0:
        return x1, y1, x2, y2

    return _sort_xyxy(
        _clamp01(x1 / float(img_w)),
        _clamp01(y1 / float(img_h)),
        _clamp01(x2 / float(img_w)),
        _clamp01(y2 / float(img_h)),
    )


def _find_first_key_ci(meta: Dict[str, Any], keys: Sequence[str]) -> Any:  # noqa: ANN401
    """Case-insensitive lookup for the first matching key."""
    lower_map = {k.lower(): k for k in meta.keys() if isinstance(k, str)}
    for key in keys:
        lk = key.lower()
        if lk in lower_map:
            return meta[lower_map[lk]]
    return None


def extract_pixel_center_box(
    meta: Optional[Dict[str, Any]],
    img_w: int,
    img_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Extract a pixel/normalized bbox from meta if present.

    IMPORTANT:
        - This function must NOT use SkyScript geo box meta['box'] (lon/lat).
        - We only accept keys meant to store pixel ROI or normalized ROI.

    Returns:
        normalized xyxy in [0,1], or None if not found/invalid.
    """
    if not meta or not isinstance(meta, dict):
        return None

    key_candidates = [
        "center_box",
        "center_bbox",
        "focus_box",
        "focus_bbox",
        "pixel_bbox",
        "pixel_box",
        "bbox_pixel",
        "roi",
        "object_bbox",
        "object_box",
        "bbox_xyxy",
    ]

    candidate = _find_first_key_ci(meta, key_candidates)
    if candidate is None:
        for val in meta.values():
            if isinstance(val, dict):
                candidate = _find_first_key_ci(val, key_candidates)
                if candidate is not None:
                    break

    if candidate is None:
        return None

    xyxy = _try_parse_xyxy(candidate)
    if xyxy is None:
        xywh = _try_parse_xywh(candidate)
        if xywh is None:
            return None
        x, y, w, h = xywh
        xyxy = (x, y, x + w, y + h)

    xyxy = _sort_xyxy(*xyxy)

    # Reject lon/lat geo bbox that looks like SkyScript meta['box'].
    if _looks_like_latlon_box(xyxy):
        return None

    return _normalize_xyxy(xyxy, img_w=img_w, img_h=img_h)


def make_center_box(
    *,
    meta: Optional[Dict[str, Any]],
    img_w: int,
    img_h: int,
    policy: str,
    center_frac: float = 0.5,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Build a normalized center box according to a policy.

    Policies:
        - "none": return None
        - "center": use a centered square box of side = center_frac (normalized)
        - "meta_pixel": try extract pixel/normalized bbox from meta (NOT geo box)

    Returns:
        normalized xyxy, or None.
    """
    pol = str(policy).lower().strip()

    if pol == "none":
        return None

    if pol == "center":
        frac = float(center_frac)
        frac = 0.1 if frac < 0.1 else 0.95 if frac > 0.95 else frac
        half = frac / 2.0
        return (0.5 - half, 0.5 - half, 0.5 + half, 0.5 + half)

    if pol == "meta_pixel":
        return extract_pixel_center_box(meta, img_w=img_w, img_h=img_h)

    raise ValueError(
        f"Unknown center_box policy: {policy}. Use one of: none|center|meta_pixel"
    )


# -----------------------------------------------------------------------------
# Meta path inference for SkyScript folder naming (images2 -> meta2)
# -----------------------------------------------------------------------------
def infer_meta_paths(img_path: str) -> List[str]:
    """
    Infer possible meta paths given an image path.

    Examples:
        .../images2/abc.jpg  -> .../meta2/abc.pickle/.pkl/.json
        .../data/images/...  -> .../data/meta/... (fallback)
        same folder fallback.

    Returns:
        A unique list of candidate meta file paths.
    """
    p = Path(str(img_path))
    stem = p.stem
    parts = list(p.parts)

    candidates: List[Path] = []

    # Map imagesX -> metaX
    for i, seg in enumerate(parts):
        match = _IMAGES_SEG_RE.match(seg)
        if match:
            x = match.group(1)
            parts2 = parts.copy()
            parts2[i] = f"meta{x}"
            meta_dir = Path(*parts2[:-1])
            for ext in _DEFAULT_META_EXTS:
                candidates.append(meta_dir / f"{stem}{ext}")
            break

    # Heuristic: replace ".../images/..." -> ".../meta/..."
    posix = str(p).replace("\\", "/")
    swapped = re.sub(r"images", "meta", posix, flags=re.IGNORECASE)
    swapped_dir = Path(swapped).parent
    for ext in _DEFAULT_META_EXTS:
        candidates.append(swapped_dir / f"{stem}{ext}")

    # Same folder fallback
    for ext in _DEFAULT_META_EXTS:
        candidates.append(p.parent / f"{stem}{ext}")

    # Unique list
    uniq: List[str] = []
    seen: set[str] = set()
    for c in candidates:
        s = str(c)
        if s not in seen:
            uniq.append(s)
            seen.add(s)

    return uniq


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SkyScriptSample:
    """A single sample returned by the dataset (before collate)."""

    image: torch.Tensor
    text: str
    center_box: Optional[Tuple[float, float, float, float]]
    path: str
    idx: int


class SkyScriptRetrievalDataset(Dataset):
    """
    CSV columns:
        - filepath: relative path, e.g. "images2/a123.jpg"
        - caption_field: e.g. "title_multi_objects"

    Notes:
        - Meta is loaded by inferring meta path from image path.
        - SkyScript meta['box'] is lon/lat => not used for pixel bboxes.
    """

    def __init__(
        self,
        data_root: str,
        csv_path: str,
        caption_field: str = "title_multi_objects",
        *,
        transform: Optional[Any] = None,
        center_box_policy: str = "none",  # none|center|meta_pixel
        center_box_frac: float = 0.5,
        on_error: str = "raise",  # raise|skip
        max_rows: int = 0,
    ) -> None:
        self.data_root = os.path.abspath(str(data_root))
        self.csv_path = str(csv_path)
        self.caption_field = str(caption_field)
        self.transform = transform
        self.center_box_policy = str(center_box_policy).lower().strip()
        self.center_box_frac = float(center_box_frac)

        if on_error not in {"raise", "skip"}:
            raise ValueError("on_error must be 'raise' or 'skip'")
        self.on_error = on_error

        usecols = ["filepath", self.caption_field]
        df = pd.read_csv(self.csv_path, usecols=usecols, keep_default_na=False)

        if "filepath" not in df.columns:
            raise ValueError("CSV must contain column 'filepath'")
        if self.caption_field not in df.columns:
            raise ValueError(f"CSV must contain caption_field '{self.caption_field}'")

        if max_rows and int(max_rows) > 0:
            df = df.iloc[: int(max_rows)].copy()

        self._filepaths = df["filepath"].astype(str).tolist()
        self._captions = df[self.caption_field].astype(str).tolist()

    def __len__(self) -> int:
        return len(self._filepaths)

    def _resolve_path(self, fp: str) -> str:
        fp = str(fp).strip()
        if not fp:
            return ""
        if os.path.isabs(fp):
            return fp
        return os.path.join(self.data_root, fp)

    @staticmethod
    def _clean_caption(val: Any) -> str:  # noqa: ANN401
        s = str(val).strip()
        return "" if (not s or s.lower() == "nan") else s

    @staticmethod
    def _load_image(img_path: str) -> Tuple[Image.Image, int, int]:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        return img, w, h

    @staticmethod
    def _try_load_meta(img_path: str) -> Optional[Dict[str, Any]]:
        for meta_path in infer_meta_paths(img_path):
            if not os.path.isfile(meta_path):
                continue
            try:
                if meta_path.endswith(".json"):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                    return obj if isinstance(obj, dict) else {"_raw": obj}

                with open(meta_path, "rb") as f:
                    obj = pickle.load(f)
                return obj if isinstance(obj, dict) else {"_raw": obj}
            except Exception:
                continue
        return None

    def _get_item_raise(self, idx: int) -> SkyScriptSample:
        fp = self._filepaths[idx]
        img_path = self._resolve_path(fp)

        if not img_path or not os.path.isfile(img_path):
            raise FileNotFoundError(f"Row {idx}: image not found: {img_path}")

        caption = self._clean_caption(self._captions[idx])
        img_pil, w, h = self._load_image(img_path)
        meta = self._try_load_meta(img_path)

        center_box = make_center_box(
            meta=meta,
            img_w=w,
            img_h=h,
            policy=self.center_box_policy,
            center_frac=self.center_box_frac,
        )

        if self.transform is None:
            raise TypeError(
                "transform must be provided and return torch.Tensor, "
                "e.g. build_skyscript_transform(...)."
            )

        img_tensor = self.transform(img_pil)
        if not isinstance(img_tensor, torch.Tensor):
            raise TypeError("transform(img) must return a torch.Tensor")

        return SkyScriptSample(
            image=img_tensor,
            text=caption,
            center_box=center_box,
            path=img_path,
            idx=int(idx),
        )

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Returns:
            Dict[str, Any] matching collate expectations, or None if on_error='skip'.
        """
        if self.on_error == "raise":
            sample = self._get_item_raise(idx)
            return {
                "image": sample.image,
                "text": sample.text,
                "center_box": sample.center_box,
                "path": sample.path,
                "idx": sample.idx,
            }

        try:
            sample = self._get_item_raise(idx)
            return {
                "image": sample.image,
                "text": sample.text,
                "center_box": sample.center_box,
                "path": sample.path,
                "idx": sample.idx,
            }
        except Exception:
            return None


# -----------------------------------------------------------------------------
# Collate utilities
# -----------------------------------------------------------------------------
def _pad_2d_int(
    seqs: List[List[int]],
    pad_id: int,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """
    Pad variable-length sequences to (B, T).

    Returns:
        input_ids: (B, T)
        lengths: (B,)
        attn_mask: (B, T) with 1 for valid positions
    """
    bsz = len(seqs)
    if bsz == 0:
        empty = torch.empty((0, 0), dtype=torch.long)
        return empty, torch.empty((0,), dtype=torch.long), empty

    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    tmax = int(lengths.max().item()) if lengths.numel() > 0 else 0

    input_ids = torch.full((bsz, tmax), fill_value=int(pad_id), dtype=torch.long)
    attn_mask = torch.zeros((bsz, tmax), dtype=torch.long)

    for i, seq in enumerate(seqs):
        if not seq:
            continue
        n = len(seq)
        input_ids[i, :n] = torch.tensor(seq, dtype=torch.long)
        attn_mask[i, :n] = 1

    return input_ids, lengths, attn_mask


def collate_retrieval(
    batch: List[Optional[Dict[str, Any]]],
    vocab: Any,
    max_len: int,
    *,
    return_raw: bool = False,
) -> Dict[str, Any]:
    """
    Collate function for retrieval training.

    Args:
        batch: List of dataset items (some can be None if on_error='skip').
        vocab: Vocab object providing encode() and pad_id/eos_id.
        max_len: Max tokens (excluding BOS/EOS). Real max becomes max_len+2.
        return_raw: Whether to include raw_texts/raw_paths.

    Returns:
        Dict containing tensors for model forward.
    """
    # Local import helps avoid circular import when tokenizer depends on datasets.
    from ..tokenizer import simple_tokenize  # pylint: disable=import-outside-toplevel

    max_len = int(max_len)
    if max_len < 1:
        raise ValueError(f"max_len must be >= 1, got {max_len}")

    items: List[Dict[str, Any]] = [b for b in batch if b is not None]
    if not items:
        out: Dict[str, Any] = {
            "images": torch.empty((0, 3, 0, 0), dtype=torch.float32),
            "input_ids": torch.empty((0, 0), dtype=torch.long),
            "lengths": torch.empty((0,), dtype=torch.long),
            "attn_mask": torch.empty((0, 0), dtype=torch.long),
            "center_boxes": torch.empty((0, 4), dtype=torch.float32),
        }
        if return_raw:
            out["raw_texts"] = []
            out["raw_paths"] = []
        return out

    imgs = [it["image"] for it in items]
    if not all(isinstance(x, torch.Tensor) for x in imgs):
        raise TypeError(
            "All item['image'] must be torch.Tensor. "
            "Provide a transform that returns torch.Tensor."
        )

    images = torch.stack(imgs, dim=0).to(dtype=torch.float32)

    hard_max = max_len + 2  # BOS/EOS
    seqs: List[List[int]] = []
    raw_texts: List[str] = []
    raw_paths: List[str] = []

    for it in items:
        text = str(it.get("text", "") or "")
        path = str(it.get("path", "") or "")

        raw_texts.append(text)
        raw_paths.append(path)

        tokens = simple_tokenize(text)[:max_len]
        ids = vocab.encode(tokens, add_bos_eos=True)

        if len(ids) > hard_max:
            ids = ids[:hard_max]
            if hasattr(vocab, "eos_id") and ids:
                ids[-1] = int(vocab.eos_id)

        seqs.append(ids)

    input_ids, lengths, attn_mask = _pad_2d_int(seqs, pad_id=int(vocab.pad_id))

    center_boxes = torch.full((len(items), 4), -1.0, dtype=torch.float32)
    for i, it in enumerate(items):
        cb = it.get("center_box")
        if cb is None:
            continue
        try:
            x1, y1, x2, y2 = cb
            center_boxes[i] = torch.tensor(
                [float(x1), float(y1), float(x2), float(y2)],
                dtype=torch.float32,
            )
        except Exception:
            continue

    out = {
        "images": images,
        "input_ids": input_ids,
        "lengths": lengths,
        "attn_mask": attn_mask,
        "center_boxes": center_boxes,
    }

    if return_raw:
        out["raw_texts"] = raw_texts
        out["raw_paths"] = raw_paths

    return out


__all__ = [
    "SkyScriptRetrievalDataset",
    "build_skyscript_transform",
    "collate_retrieval",
]

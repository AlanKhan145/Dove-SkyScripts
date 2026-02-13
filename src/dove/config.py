# src/dove/config.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, TypeAlias, Union

CaptionField: TypeAlias = Literal["title_multi_objects", "title"]


def _as_path(p: Optional[Union[str, Path]]) -> Optional[Path]:
    if p is None:
        return None
    if isinstance(p, str) and p.strip() == "":
        return None
    return p if isinstance(p, Path) else Path(p)


def _deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)  # type: ignore[arg-type]
        else:
            base[k] = v
    return base


def _filter_dataclass_kwargs(cls: type, raw: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {f.name for f in dataclass_fields(cls)}
    return {k: v for k, v in raw.items() if k in allowed}


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


@dataclass(frozen=True)
class DataConfig:
    image_size: int = 256
    caption_field: CaptionField = "title_multi_objects"
    use_meta_center_box: bool = True

    def validate(self) -> None:
        if self.image_size <= 0:
            raise ValueError(f"data.image_size must be > 0, got {self.image_size}")
        if self.caption_field not in ("title_multi_objects", "title"):
            raise ValueError(f"data.caption_field invalid: {self.caption_field}")


@dataclass(frozen=True)
class TextConfig:
    max_len: int = 40
    min_freq: int = 2

    glove_path: Optional[Path] = None
    glove_dim: int = 300
    word_emb_dim: int = 300

    gru_hidden: int = 256
    attn_dim: int = 256

    dropout: float = 0.0
    dtga_mlp_ratio: float = 1.0

    def validate(self) -> None:
        if self.max_len <= 0:
            raise ValueError(f"text.max_len must be > 0, got {self.max_len}")
        if self.min_freq < 1:
            raise ValueError(f"text.min_freq must be >= 1, got {self.min_freq}")

        if self.glove_dim <= 0 or self.word_emb_dim <= 0:
            raise ValueError("text.glove_dim and text.word_emb_dim must be > 0")

        if self.gru_hidden <= 0 or self.attn_dim <= 0:
            raise ValueError("text.gru_hidden and text.attn_dim must be > 0")

        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"text.dropout must be in [0,1), got {self.dropout}")

        if self.dtga_mlp_ratio <= 0:
            raise ValueError(f"text.dtga_mlp_ratio must be > 0, got {self.dtga_mlp_ratio}")

        gp = _as_path(self.glove_path)
        if gp is not None and not gp.exists():
            raise FileNotFoundError(f"text.glove_path not found: {gp}")


@dataclass(frozen=True)
class VisualConfig:
    pretrained_backbone: bool = True
    freeze_backbone: bool = True
    freeze_bn_stats: bool = True

    roi_out: int = 3
    roi_grid: int = 6
    add_center_box: bool = True
    roi_backend: str = "auto"

    d_model: int = 512

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"visual.d_model must be > 0, got {self.d_model}")
        if self.roi_out <= 0:
            raise ValueError(f"visual.roi_out must be > 0, got {self.roi_out}")
        if self.roi_grid <= 0:
            raise ValueError(f"visual.roi_grid must be > 0, got {self.roi_grid}")


@dataclass(frozen=True)
class FusionConfig:
    enable_roam: bool = True
    n_heads: int = 4
    roam_layers: int = 1
    dropout: float = 0.0

    def validate(self) -> None:
        if self.n_heads <= 0:
            raise ValueError(f"fusion.n_heads must be > 0, got {self.n_heads}")
        if self.roam_layers <= 0:
            raise ValueError(f"fusion.roam_layers must be > 0, got {self.roam_layers}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"fusion.dropout must be in [0,1), got {self.dropout}")


@dataclass(frozen=True)
class LossConfig:
    margin: float = 0.2
    lambda_g: float = 10.0

    use_temperature: bool = False
    temperature: float = 0.07

    use_logit_scale: bool = False

    def validate(self) -> None:
        if self.margin < 0:
            raise ValueError(f"loss.margin must be >= 0, got {self.margin}")
        if self.lambda_g < 0:
            raise ValueError(f"loss.lambda_g must be >= 0, got {self.lambda_g}")
        if self.use_temperature and self.temperature <= 0:
            raise ValueError(f"loss.temperature must be > 0 when use_temperature=True, got {self.temperature}")


@dataclass(frozen=True)
class TrainingConfig:
    out_dir: Path = Path("runs/dove_skyscript")
    epochs: int = 20
    batch_size: int = 64

    lr: float = 2e-4
    weight_decay: float = 0.05

    amp: bool = True
    num_workers: int = 4
    seed: int = 42
    grad_clip: Optional[float] = 1.0

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"train.epochs must be > 0, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"train.batch_size must be > 0, got {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"train.lr must be > 0, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"train.weight_decay must be >= 0, got {self.weight_decay}")
        if self.num_workers < 0:
            raise ValueError(f"train.num_workers must be >= 0, got {self.num_workers}")
        if self.seed < 0:
            raise ValueError(f"train.seed must be >= 0, got {self.seed}")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError(f"train.grad_clip must be > 0 if set, got {self.grad_clip}")


@dataclass(frozen=True)
class RuntimeConfig:
    return_aux: bool = True
    validate_on_init: bool = True
    prefetch_factor: int = 2

    def validate(self) -> None:
        if self.prefetch_factor <= 0:
            raise ValueError(f"runtime.prefetch_factor must be > 0, got {self.prefetch_factor}")


@dataclass
class DOVEConfig:
    data: DataConfig = field(default_factory=DataConfig)
    text: TextConfig = field(default_factory=TextConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    embed_dim: int = 512

    def __post_init__(self) -> None:
        if self.runtime.validate_on_init:
            self.validate()

    def validate(self) -> None:
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {self.embed_dim}")

        self.data.validate()
        self.text.validate()
        self.visual.validate()
        self.fusion.validate()
        self.loss.validate()
        self.train.validate()
        self.runtime.validate()

        if self.visual.d_model != self.embed_dim:
            raise ValueError(
                "Shape invariant failed: expected visual.d_model == embed_dim "
                f"(got visual.d_model={self.visual.d_model}, embed_dim={self.embed_dim})."
            )

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(asdict(self))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DOVEConfig":
        dd = dict(d)

        data = DataConfig(**_filter_dataclass_kwargs(DataConfig, dd.get("data", {})))

        text_raw = dict(dd.get("text", {}))
        text_raw = _filter_dataclass_kwargs(TextConfig, text_raw)
        if "glove_path" in text_raw:
            text_raw["glove_path"] = _as_path(text_raw["glove_path"])
        text = TextConfig(**text_raw)

        vis_raw = dict(dd.get("visual", {}))
        # backward-compat: roi_grid may be tuple/list (6,6) => take first
        if "roi_grid" in vis_raw and isinstance(vis_raw["roi_grid"], (list, tuple)):
            vis_raw["roi_grid"] = int(vis_raw["roi_grid"][0])
        vis_raw = _filter_dataclass_kwargs(VisualConfig, vis_raw)
        visual = VisualConfig(**vis_raw)

        fusion = FusionConfig(**_filter_dataclass_kwargs(FusionConfig, dd.get("fusion", {})))
        loss = LossConfig(**_filter_dataclass_kwargs(LossConfig, dd.get("loss", {})))

        train_raw = dict(dd.get("train", {}))
        train_raw = _filter_dataclass_kwargs(TrainingConfig, train_raw)
        if "out_dir" in train_raw:
            train_raw["out_dir"] = Path(train_raw["out_dir"])
        train = TrainingConfig(**train_raw)

        runtime = RuntimeConfig(**_filter_dataclass_kwargs(RuntimeConfig, dd.get("runtime", {})))

        embed_dim = int(dd.get("embed_dim", 512))

        return cls(
            data=data,
            text=text,
            visual=visual,
            fusion=fusion,
            loss=loss,
            train=train,
            runtime=runtime,
            embed_dim=embed_dim,
        )

    def to_json(self, path: Union[str, Path], indent: int = 2) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=indent), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "DOVEConfig":
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def with_overrides(self, overrides: Mapping[str, Any]) -> "DOVEConfig":
        merged = _deep_update(self.to_dict(), overrides)
        return DOVEConfig.from_dict(merged)


__all__ = [
    "CaptionField",
    "DataConfig",
    "TextConfig",
    "VisualConfig",
    "FusionConfig",
    "LossConfig",
    "TrainingConfig",
    "RuntimeConfig",
    "DOVEConfig",
]

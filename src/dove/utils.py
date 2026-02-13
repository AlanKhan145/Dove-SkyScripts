"""
Small, production-friendly utilities used by DOVE train/eval.

Includes:
- Reproducibility: set_seed(seed)
- Filesystem: ensure_dir(path)
- Model stats: count_params, count_trainable_params
- Math: l2norm for embedding normalization
- Checkpointing: safe_torch_save / safe_torch_load (atomic save)
- Model Ops: freeze_batch_norm_2d, replace_linear (adapted from OpenCLIP)
- Helpers: to_2tuple, to_ntuple
"""

from __future__ import annotations

import collections.abc
import os
import random
from itertools import repeat
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import numpy as np  # optional dependency
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

import torch
import torch.nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d


# -----------------------------------------------------------------------------
# Reproducibility & Filesystem
# -----------------------------------------------------------------------------

def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """
    Seed Python, PyTorch, and NumPy (if available).

    Args:
        seed: non-negative integer seed.
        deterministic: if True, toggles cuDNN deterministic settings.
            This improves reproducibility but may reduce speed.
    """
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    if np is not None:
        np.random.seed(seed)  # type: ignore[union-attr]

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists. Safe for concurrent calls.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------------------------------------------------------
# Model Statistics
# -----------------------------------------------------------------------------

def count_params(model: torch.nn.Module) -> int:
    """Total number of parameters (trainable + frozen)."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model: torch.nn.Module) -> int:
    """Number of parameters with requires_grad=True."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# Math / Tensors
# -----------------------------------------------------------------------------

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    L2-normalize tensor along `dim` with numerical stability.
    
    This implementation is stable for fp16/bf16 by computing norm in fp32.
    """
    if x.numel() == 0:
        return x

    if x.dtype in (torch.float16, torch.bfloat16):
        denom = (
            torch.linalg.norm(x.float(), ord=2, dim=dim, keepdim=True)
            .clamp_min(eps)
            .to(dtype=x.dtype)
        )
    else:
        denom = torch.linalg.norm(x, ord=2, dim=dim, keepdim=True).clamp_min(eps)

    return x / denom


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------

def safe_torch_save(obj: Any, path: Union[str, Path]) -> Path:
    """
    Atomically save with torch.save (temp file + replace) to avoid partial writes.
    """
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(p)
    return p


def safe_torch_load(
    path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = "cpu",
) -> Any:
    """
    Safe torch.load wrapper with default map_location='cpu'.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return torch.load(p, map_location=map_location)


# -----------------------------------------------------------------------------
# Model Operations (Adapted from OpenCLIP)
# -----------------------------------------------------------------------------

def freeze_batch_norm_2d(
    module: nn.Module, 
    module_match: Dict[str, Any] = {}, 
    name: str = ''
) -> nn.Module:
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`.
    
    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module with frozen batch norms.
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
        
    if is_match and isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


def replace_linear(
    model: nn.Module, 
    linear_replacement: Callable[..., nn.Module], 
    include_modules: List[str] = ['c_fc', 'c_proj'], 
    copy_weights: bool = True
) -> nn.Module:
    """
    Replaces specified linear layers with a replacement module (e.g., for int8 or LoRA).
    Recursively traverses the model.
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear(module, linear_replacement, include_modules, copy_weights)

        if isinstance(module, torch.nn.Linear) and name in include_modules:
            old_module = model._modules[name]
            # Assumes replacement signature matches (in, out, bias)
            model._modules[name] = linear_replacement(
                old_module.in_features,
                old_module.out_features,
                bias=old_module.bias is not None,
            )
            if copy_weights:
                model._modules[name].weight.data.copy_(old_module.weight.data)
                if model._modules[name].bias is not None:
                    model._modules[name].bias.data.copy_(old_module.bias.data)

    return model


def convert_int8_model_to_inference_mode(model: nn.Module) -> None:
    """
    Prepares int8 layers (if they have a prepare_for_eval method) for inference.
    """
    for m in model.modules():
        if hasattr(m, 'prepare_for_eval'):
            int8_original_dtype = getattr(m.weight, 'dtype', None)
            m.prepare_for_eval()
            if int8_original_dtype is not None:
                m.int8_original_dtype = int8_original_dtype


# -----------------------------------------------------------------------------
# Tuple Helpers (From PyTorch internals/OpenCLIP)
# -----------------------------------------------------------------------------

def _ntuple(n: int) -> Callable[[Any], Tuple[Any, ...]]:
    def parse(x: Any) -> Tuple[Any, ...]:
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)


__all__ = [
    # Reproducibility & System
    "set_seed",
    "ensure_dir",
    
    # Model Stats
    "count_params",
    "count_trainable_params",
    
    # Math
    "l2norm",
    
    # Checkpointing
    "safe_torch_save",
    "safe_torch_load",
    
    # Model Ops (OpenCLIP adaptations)
    "freeze_batch_norm_2d",
    "replace_linear",
    "convert_int8_model_to_inference_mode",
    
    # Tuple Helpers
    "to_1tuple",
    "to_2tuple",
    "to_3tuple",
    "to_4tuple",
    "to_ntuple",
]
"""
DOVE package public API.

Design goals:
- Keep import-time lightweight (avoid importing torch/torchvision unless needed).
- Expose a small, stable surface area used by training/eval scripts.
- Provide lazy attribute resolution for heavier modules (PEP 562).
- Cache resolved symbols in module globals for subsequent fast access.

Notes:
- This file should not import torch at import-time.
- All heavy imports are done lazily inside __getattr__.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    # Submodules
    "config",
    "datasets",
    "models",
    "losses",
    "tokenizer",
    "utils",
    
    # Public symbols (stable API)
    # Config
    "DOVEConfig",
    
    # Tokenizer (Updated for CLIP BPE)
    "tokenize",
    "decode",
    "SimpleTokenizer",
    
    # Datasets
    "SkyScriptRetrievalDataset",
    "build_skyscript_transform",
    "collate_retrieval",
    
    # Models
    "DOVEModel",
    
    # Losses
    "DOVELoss",
]


# -----------------------------------------------------------------------------
# Lazy resolution maps
# -----------------------------------------------------------------------------
# Lazy submodules (e.g., `import dove; dove.datasets`)
_LAZY_MODULES: dict[str, str] = {
    "config": "dove.config",
    "datasets": "dove.datasets",
    "models": "dove.models",
    "losses": "dove.losses",
    "tokenizer": "dove.tokenizer",
    "utils": "dove.utils",
}

# Lazy attributes (e.g., `from dove import DOVEConfig`)
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Config
    "DOVEConfig": ("dove.config", "DOVEConfig"),
    
    # Tokenizer
    "tokenize": ("dove.tokenizer", "tokenize"),
    "decode": ("dove.tokenizer", "decode"),
    "SimpleTokenizer": ("dove.tokenizer", "SimpleTokenizer"),
    
    # Datasets
    "SkyScriptRetrievalDataset": ("dove.datasets", "SkyScriptRetrievalDataset"),
    "build_skyscript_transform": ("dove.datasets", "build_skyscript_transform"),
    "collate_retrieval": ("dove.datasets", "collate_retrieval"),
    
    # Models
    "DOVEModel": ("dove.models", "DOVEModel"),
    
    # Losses
    "DOVELoss": ("dove.losses", "DOVELoss"),
}


def __getattr__(name: str) -> Any:
    """
    PEP 562: lazily resolve selected attributes and submodules.

    Behavior:
    - If `name` is a lazy submodule, import it and cache it in globals().
    - If `name` is a lazy symbol, import its module, extract the symbol,
      and cache it in globals().
    - Otherwise raise AttributeError.
    """
    # Lazy submodules (e.g., dove.datasets)
    if name in _LAZY_MODULES:
        module = import_module(_LAZY_MODULES[name])
        globals()[name] = module  # cache
        return module

    # Lazy public symbols
    if name in _LAZY_ATTRS:
        mod_name, attr_name = _LAZY_ATTRS[name]
        module = import_module(mod_name)
        value = getattr(module, attr_name)
        globals()[name] = value  # cache
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """
    Return enhanced dir() listing including lazily-resolved names.

    This is helpful for interactive usage and IDE autocompletion.
    """
    return sorted(set(globals()) | set(_LAZY_MODULES) | set(_LAZY_ATTRS))


if TYPE_CHECKING:
    # For type checkers only; does not execute at runtime import-time.
    from .config import DOVEConfig
    from .datasets import (
        SkyScriptRetrievalDataset,
        build_skyscript_transform,
        collate_retrieval,
    )
    from .losses import DOVELoss
    from .models import DOVEModel
    from .tokenizer import SimpleTokenizer, tokenize, decode
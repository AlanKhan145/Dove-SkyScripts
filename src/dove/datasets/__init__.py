"""
File: src/dove/datasets/__init__.py
Purpose:
    Public exports for dataset components (SkyScript retrieval dataset, transforms,
    and collate function) so training scripts can import consistently.
"""

from __future__ import annotations

from .skyscript import (
    SkyScriptRetrievalDataset,
    build_skyscript_transform,
    collate_retrieval,
)

__all__ = [
    "SkyScriptRetrievalDataset",
    "build_skyscript_transform",
    "collate_retrieval",
]

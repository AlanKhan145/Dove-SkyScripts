"""
File: src/dove/utils/run_logging.py
Purpose:
    Small utilities for training-time logging:

    - Setup a unified logger (console + file).
    - Tee stdout/stderr into a file for reproducible training logs.
    - JSON/JSONL writers for step/epoch metrics.
    - Environment dump (torch/cuda/platform versions).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional


def now_str() -> str:
    """Return local time string for logging."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def save_json(path: str, obj: Dict[str, Any]) -> None:
    """Write a JSON file (pretty)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    """Append one record to a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def setup_logger(out_dir: str, name: str = "dove_train") -> logging.Logger:
    """Create logger that writes both to console and out_dir/run.log."""
    os.makedirs(out_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    fh = logging.FileHandler(os.path.join(out_dir, "run.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class Tee:
    """Duplicate writes to multiple streams."""

    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                continue

    def flush(self) -> None:
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                continue


@contextmanager
def tee_stdout_stderr(log_path: str):
    """Context manager: tee stdout/stderr to a file."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = Tee(old_out, f)  # type: ignore[assignment]
        sys.stderr = Tee(old_err, f)  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout = old_out
            sys.stderr = old_err


def dump_env(out_dir: str) -> Dict[str, Any]:
    """Collect and save environment info to out_dir/env.json."""
    try:
        import torch
    except Exception:
        torch = None  # type: ignore

    info: Dict[str, Any] = {
        "time": now_str(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }

    if torch is not None:
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_count"] = int(torch.cuda.device_count())
            try:
                info["cuda_name0"] = torch.cuda.get_device_name(0)
            except Exception:
                pass

    save_json(os.path.join(out_dir, "env.json"), info)
    return info


def get_lr(opt: Any) -> float:
    """Safely read LR from optimizer."""
    try:
        return float(opt.param_groups[0]["lr"])
    except Exception:
        return 0.0

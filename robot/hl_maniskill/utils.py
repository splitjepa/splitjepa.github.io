"""I/O helpers shared by the experiment."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_default)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def default_device(device: str | None) -> str:
    if device is not None:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_run_dir(exp_root: Path, stage: str, tag: str) -> Path:
    """Create a unique output dir: <exp_root>/<stage>/<tag>_<timestamp>."""
    import datetime

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = exp_root / stage / f"{tag}_{ts}"
    d.mkdir(parents=True, exist_ok=False)
    return d

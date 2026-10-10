"""Configuration system."""

from __future__ import annotations
import copy
from pathlib import Path
from typing import Any
import yaml


class Config(dict):
    """Dict with attribute access for convenience."""
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __deepcopy__(self, memo):
        return Config(copy.deepcopy(dict(self), memo))


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _to_config(obj):
    """Recursively convert plain dicts into Config."""
    if isinstance(obj, dict):
        return Config({k: _to_config(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config(v) for v in obj]
    return obj


def _apply_dotted(cfg: dict, key: str, value: str) -> None:
    """Set cfg['a']['b']['c'] = value from a dotted key 'a.b.c'."""
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    # attempt typed parsing
    raw: Any = value
    if value.lower() in ("true", "false"):
        raw = value.lower() == "true"
    else:
        for cast in (int, float):
            try:
                raw = cast(value)
                break
            except ValueError:
                pass
    node[parts[-1]] = raw


def load_config(
    config_files: list[str],
    overrides: list[str] | None = None,
    root: str | None = None,
) -> Config:
    """Load and merge YAML config files."""
    merged: dict = {}
    for cf in config_files:
        path = Path(cf)
        data = _load_yaml(path)
        # resolve inheritance
        inherited = data.pop("inherit", None)
        if inherited:
            for inc in inherited:
                inc_path = Path(inc)
                if not inc_path.is_absolute():
                    inc_path = path.parent / inc_path
                merged = _deep_merge(merged, _load_yaml(inc_path))
        merged = _deep_merge(merged, data)

    if overrides:
        for ov in overrides:
            key, _, value = ov.partition("=")
            if not value:
                raise ValueError(f"override must be key=value, got: {ov}")
            _apply_dotted(merged, key.strip(), value.strip())

    if root is not None:
        merged["root"] = str(Path(root).resolve())

    # ensure commonly used absolute paths are derived
    root_path = Path(merged.get("root", ".")).resolve()
    merged.setdefault("paths", {})
    merged["paths"].setdefault("data_root", str(root_path / "data"))
    merged["paths"].setdefault("exp_root", str(root_path / "experiments"))

    return Config(_to_config(merged))

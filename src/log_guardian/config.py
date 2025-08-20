"""
Log Guardian configuration loader.

Supports JSON or YAML configs. Also merges in environment variable overrides
(prefixed with LOG_GUARDIAN_).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # optional
except ImportError:
    yaml = None


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load configuration dict from file (JSON or YAML).
    Falls back to empty dict if no path is given.
    Environment variables prefixed with LOG_GUARDIAN_ override keys.
    """
    cfg: Dict[str, Any] = {}

    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        if p.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise ImportError("PyYAML is required for YAML configs")
            with p.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            with p.open("r", encoding="utf-8") as f:
                cfg = json.load(f)

    # Merge in environment overrides
    for k, v in os.environ.items():
        if k.startswith("LOG_GUARDIAN_"):
            key = k.removeprefix("LOG_GUARDIAN_").lower()
            cfg[key] = v

    return cfg


def get(key: str, default: Any = None, *, config: Dict[str, Any] | None = None) -> Any:
    """
    Fetch a single config value (case-insensitive).
    """
    if config is None:
        config = {}
    return config.get(key.lower(), default)

"""Utility helpers for Log Guardian."""
from __future__ import annotations

import hashlib
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict


def sha256_of_file(path: str | Path, block_size: int = 65536) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    p = Path(path)
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


essure_dir_doc = """Ensure directory exists and return as Path."""

def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists and return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load JSON file safely."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str | Path, indent: int = 2) -> None:
    """Save dict as JSON."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def json_record_to_str(record: logging.LogRecord) -> str:
    """Convert a LogRecord to a JSON string."""
    payload: Dict[str, Any] = {
        "timestamp": record.created,
        "level": record.levelname,
        "name": record.name,
        "message": record.getMessage(),
        "module": record.module,
        "filename": record.filename,
        "lineno": record.lineno,
    }
    if record.exc_info:
        # record.exc_info is a tuple (exc_type, exc_value, exc_traceback)
        payload["exc_info"] = "".join(traceback.format_exception(*record.exc_info))
    # Some code sets custom attributes; guard and merge if present
    if hasattr(record, "extra") and isinstance(getattr(record, "extra"), dict):
        payload.update(record.extra)  # type: ignore[arg-type]
    return json.dumps(payload, ensure_ascii=False, default=str)


def get_config() -> Dict[str, Any]:
    """Load the main configuration dict via :mod:`src.log_guardian.config`.

    Wrapper to avoid name shadowing with this module. Raises ValueError if
    the loader returns an empty mapping.
    """
    from .config import load_config as _load_config

    cfg = _load_config()
    if not cfg:
        raise ValueError("No configuration loaded")
    return cfg

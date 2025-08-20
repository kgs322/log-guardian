"""Centralized logging setup for Log Guardian.

Features:
- Sensible defaults (INFO level, console output)
- Optional JSON logs via env var `LOG_GUARDIAN_LOG_JSON=1`
- Optional rotating file handler (`LOG_GUARDIAN_LOG_FILE=...`)
- Module-level helper `get_logger(name)`

Usage
-----
from .logging_setup import setup_logging, get_logger
setup_logging()  # call once at process start
log = get_logger(__name__)
log.info("hello")
"""
from __future__ import annotations

import json
import logging
import logging.config
import os
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """A minimal JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "level": record.levelname,
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Optional extras
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        return json.dumps(payload, ensure_ascii=False)


DEFAULT_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def setup_logging(
    *,
    level: str | int | None = None,
    json_logs: Optional[bool] = None,
    log_file: str | Path | None = None,
    file_max_bytes: int = 5 * 1024 * 1024,
    file_backup_count: int = 3,
) -> None:
    """Configure root logging. Safe to call multiple times (idempotent-ish)."""
    # Resolve settings from env if not provided
    if level is None:
        level = os.getenv("LOG_GUARDIAN_LOG_LEVEL", "INFO").upper()
    if json_logs is None:
        json_logs = _env_flag("LOG_GUARDIAN_LOG_JSON", False)
    if log_file is None:
        log_file = os.getenv("LOG_GUARDIAN_LOG_FILE")

    handlers: Dict[str, Any] = {}
    root_handlers: list[str] = []

    # Console handler
    handlers["console"] = {
        "class": "logging.StreamHandler",
        "level": level,
        "formatter": "json" if json_logs else "console",
        "stream": "ext://sys.stdout",
    }
    root_handlers.append("console")

    # File handler (optional rotating)
    if log_file:
        log_path = str(Path(log_file))
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": "json" if json_logs else "console",
            "filename": log_path,
            "maxBytes": file_max_bytes,
            "backupCount": file_backup_count,
            "encoding": "utf-8",
        }
        root_handlers.append("file")

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "format": DEFAULT_FMT,
                "datefmt": DEFAULT_DATEFMT,
            },
            "json": {"()": JsonFormatter},
        },
        "handlers": handlers,
        "root": {"level": level, "handlers": root_handlers},
    }

    logging.config.dictConfig(config)


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger; call `setup_logging()` once at process startup first."""
    return logging.getLogger(name if name else __name__)

# End of logging_setup.py
# ---------------------------------------------------------------------
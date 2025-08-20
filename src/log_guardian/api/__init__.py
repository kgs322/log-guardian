"""Log Guardian API package.

Exports the FastAPI `app` instance or the `create_app(cfg)` factory (if present),
plus the public Pydantic schemas so callers can import from a single place.

Example
-------
from src.log_guardian.api import app  # or create_app
from src.log_guardian.api import ScoreRequest, ScoreResponse
"""
from __future__ import annotations

from typing import Any, Optional

__all__ = [
    "app",
    "create_app",
    "get_app",
    # Schemas
    "HealthResponse",
    "ScoreRequest",
    "ScoreResponse",
]

# Try to expose the FastAPI app (or a factory) if the module provides them.
try:  # pragma: no cover - thin re-export wrapper
    from .app import app as app  # type: ignore
except Exception:  # pragma: no cover
    app = None  # type: ignore

try:  # pragma: no cover
    from .app import create_app as create_app  # type: ignore
except Exception:  # pragma: no cover
    create_app = None  # type: ignore


# Public schemas re-export (optional but convenient for SDK-like usage)
try:  # pragma: no cover
    from .schemas import (
        HealthResponse,
        ScoreRequest,
        ScoreResponse,
    )
except Exception:  # pragma: no cover
    # Allow importing the package even if schemas aren't ready yet during early scaffolding
    HealthResponse = ScoreRequest = ScoreResponse = None  # type: ignore


def get_app(cfg: Optional[dict[str, Any]] = None):
    """Return a FastAPI app instance.

    - If a factory `create_app(cfg)` exists, prefer it so configuration is applied.
    - Else, return the module-level `app` instance.
    - Raises a clear error if neither is available.
    """
    if create_app is not None:
        return create_app(cfg or {})  # type: ignore[misc]
    if app is not None:
        return app
    raise ImportError(
        "API not initialized: expected `app` or `create_app(cfg)` in src.log_guardian.api.app"
    )

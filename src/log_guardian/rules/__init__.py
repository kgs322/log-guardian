"""Rule-based detection package for Log Guardian.

Holds declarative signatures and simple heuristic rules that complement
ML anomaly detection. Typical usage:

from src.log_guardian.rules import signatures
alerts = signatures.apply_all(df)
"""
from __future__ import annotations

__all__ = [
    "signatures",
]

try:
    from . import signatures  # type: ignore
except Exception:
    signatures = None  # type: ignore
from .signatures import *  # noqa: F401, F403
from .signatures import (
    apply_all,
    detect_suspicious_activity,
    flag_known_patterns,
    enrich_with_signatures,
)
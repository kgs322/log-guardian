"""Log Guardian – anomaly detection & rule-based log analysis.

Subpackages:
- ingestion: parsers and schemas for raw logs
- features: feature builders for modeling
- modeling: training, prediction, and post-processing
- rules: heuristic signatures
- api: FastAPI app and schemas
"""
from __future__ import annotations

__version__ = "0.1.0"

# Keep the package lightweight: expose subpackages via __all__ without star-importing
__all__ = [
    "ingestion",
    "features",
    "modeling",
    "rules",
    "api",
]

# Optional convenience imports (safe, no heavy side effects)
try:  # pragma: no cover
    from . import ingestion, features, modeling, rules, api  # type: ignore
except Exception:  # pragma: no cover
    # Allow partial scaffolding during development
    pass

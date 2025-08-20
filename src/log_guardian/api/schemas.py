"""Pydantic schemas for the Log Guardian API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Simple health check response."""
    status: str = Field(..., description="Service status, e.g. 'ok'")

    class Config:
        extra = "ignore"


class ScoreRequest(BaseModel):
    """Request body for /score endpoint.

    - `rows` is a list of dictionaries, each representing one observation/row.
    - Extra keys are allowed so clients can include non-feature columns (e.g., ids/timestamps).
    """
    rows: List[Dict[str, Any]] = Field(..., description="List of records to score")

    class Config:
        extra = "allow"  # accept additional fields in rows without validation errors

    class ConfigDict:
        # For Pydantic v2 compatibility (harmless in v1)
        extra = "allow"

    model_config = ConfigDict  # type: ignore[assignment]

    class Config:
        # Keep v1 compatibility too
        extra = "allow"

    @staticmethod
    def example() -> "ScoreRequest":
        return ScoreRequest(
            rows=[
                {"failed_login_rate": 0.02, "unique_ports": 5, "reqs_per_min": 120, "status_4xx_5xx_ratio": 0.03, "host": "web-1"},
                {"failed_login_rate": 0.40, "unique_ports": 55, "reqs_per_min": 900, "status_4xx_5xx_ratio": 0.45, "host": "web-2"},
            ]
        )


class ScoreResponse(BaseModel):
    """Response body for /score endpoint."""
    scores: List[float] = Field(..., description="Anomaly scores aligned with request rows")
    flagged: List[int] = Field(..., description="Binary flags (1=anomalous) aligned with scores")
    # Optional metadata fields (not currently populated by app.py but reserved for future use)
    model_name: Optional[str] = Field(None, description="Model name used for scoring")
    model_version: Optional[str] = Field(None, description="Model version used for scoring")

    class Config:
        extra = "ignore"

    @staticmethod
    def example() -> "ScoreResponse":
        return ScoreResponse(
            scores=[0.01, 2.45],
            flagged=[0, 1],
            model_name="isolation_forest",
            model_version="0.2.0",
        )


# OpenAPI Examples (FastAPI reads from schema_extra if present)
HealthResponse.update_forward_refs()
ScoreRequest.update_forward_refs()
ScoreResponse.update_forward_refs()

HealthResponse.__config__ = type("Config", (), {"schema_extra": {"example": {"status": "ok"}}})
ScoreRequest.__config__ = type(
    "Config",
    (),
    {
        "schema_extra": {
            "example": {
                "rows": [
                    {"failed_login_rate": 0.02, "unique_ports": 5, "reqs_per_min": 120, "status_4xx_5xx_ratio": 0.03, "host": "web-1"},
                    {"failed_login_rate": 0.40, "unique_ports": 55, "reqs_per_min": 900, "status_4xx_5xx_ratio": 0.45, "host": "web-2"},
                ]
            }
        }
    },
)
ScoreResponse.__config__ = type(
    "Config",
    (),
    {
        "schema_extra": {
            "example": {
                "scores": [0.01, 2.45],
                "flagged": [0, 1],
                "model_name": "isolation_forest",
                "model_version": "0.2.0",
            }
        }
    },
)
__all__ = [
    "HealthResponse",
    "ScoreRequest",
    "ScoreResponse",
]
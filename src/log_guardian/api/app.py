"""FastAPI application for Log Guardian.

Provides health checks and a /score endpoint to run anomaly detection
using the latest model from the registry (or an override via env/config).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException

# ------------------------------------------------------------------
# Optional imports (keep the API resilient during scaffolding)
# ------------------------------------------------------------------
try:  # Typed schemas if available
    from .schemas import HealthResponse, ScoreRequest, ScoreResponse  # type: ignore
except Exception:  # Fallback tiny models so the service can still boot
    from pydantic import BaseModel

    class HealthResponse(BaseModel):
        status: str

    class ScoreRequest(BaseModel):
        rows: list[dict]

    class ScoreResponse(BaseModel):
        scores: list[float]
        flagged: list[int]

try:
    from ..modeling import predict  # type: ignore
except Exception:
    predict = None  # type: ignore

LOGGER = logging.getLogger("log_guardian.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ------------------------------------------------------------------
# Model loading / caching
# ------------------------------------------------------------------
LATEST_MODEL: Tuple[Any, dict] | None = None


def _registry_path() -> Path:
    return Path(os.getenv("LOG_GUARDIAN_MODEL_REGISTRY", "models/registry.json"))


def _override_model_path() -> Optional[Path]:
    val = os.getenv("LOG_GUARDIAN_MODEL_PATH")
    return Path(val) if val else None


def load_model(registry_path: str | Path | None = None,
               override_path: str | Path | None = None) -> Tuple[Any, dict]:
    """Load model from explicit override or the registry's current_model."""
    import joblib

    reg_path = Path(registry_path) if registry_path else _registry_path()
    override = Path(override_path) if override_path else _override_model_path()

    if override and override.exists():
        model = joblib.load(override)
        return model, {"name": override.stem, "version": "(override)", "path": str(override)}

    if not reg_path.exists():
        raise FileNotFoundError(f"Registry not found: {reg_path}")

    with open(reg_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    model_info = registry.get("current_model")
    if not model_info:
        raise RuntimeError("No current_model in registry")

    model_path = Path(model_info["path"])  # stored path should be repo-relative
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    return model, model_info


def get_model() -> Tuple[Any, dict]:
    global LATEST_MODEL
    if LATEST_MODEL is None:
        LATEST_MODEL = load_model()
        LOGGER.info("Loaded model: %s v%s", LATEST_MODEL[1].get("name"), LATEST_MODEL[1].get("version"))
    return LATEST_MODEL


# ------------------------------------------------------------------
# Factory & routes
# ------------------------------------------------------------------

def create_app(cfg: Optional[Dict[str, Any]] = None) -> FastAPI:
    app = FastAPI(title="Log Guardian API", version="0.1.0")

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        if predict is None or not hasattr(predict, "score"):
            raise HTTPException(status_code=500, detail="predict.score not available")

        # Convert request rows to a DataFrame
        try:
            df = pd.DataFrame(req.rows)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid rows: {e}")

        # Score
        try:
            model, _info = get_model()
            scores = predict.score(model, df)  # expected to return 1D array-like
            scores = pd.Series(scores, name="anomaly_score").astype(float)
            # Simple default flag: mean + 2*std (override in caller if needed)
            flags = (scores >= scores.mean() + 2 * scores.std()).astype(int).tolist()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")

        return ScoreResponse(scores=scores.tolist(), flagged=flags)

    return app


# Uvicorn import target
app = create_app()


def get_app(cfg: Optional[Dict[str, Any]] = None) -> FastAPI:
    """Return a FastAPI app instance (prefers factory to apply cfg)."""
    return create_app(cfg or {})
    "No FastAPI app or factory available in log_guardian.api module."
    "Ensure src/log_guardian/api/app.py defines `app` or `create_app(cfg)`."
    "If using a factory, ensure it is imported correctly."
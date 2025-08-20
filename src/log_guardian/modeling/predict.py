"""Prediction utilities for Log Guardian modeling.

Provides a unified interface to load models from the registry
and score new log-derived feature sets.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, List

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Core loading
# ---------------------------------------------------------------------

def load_model(model_path: str | Path) -> Any:
    """Load a serialized model artifact (joblib/pickle)."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


# ---------------------------------------------------------------------
# Scoring interfaces
# ---------------------------------------------------------------------

def predict(
    df: pd.DataFrame,
    *,
    model: Any,
    features: Optional[List[str]] = None,
    return_scores: bool = True,
) -> np.ndarray | Dict[str, Any]:
    """Run model prediction on a DataFrame of features.

    Parameters
    ----------
    df : pd.DataFrame
        Input feature matrix.
    model : object | dict
        Trained estimator, or an artifact dict like {"model": <estimator>, "features": [...] }.
    features : list[str], optional
        Subset of feature columns to use (ignored if artifact contains 'features').
    return_scores : bool
        If True, return a dict with {"labels", "scores"} when possible.

    Returns
    -------
    np.ndarray or dict
        Predictions or dict with {"labels", "scores"}.
    """
    # Unpack artifact dict if provided
    mdl = model.get("model", model) if isinstance(model, dict) else model
    feats = None
    if isinstance(model, dict):
        feats = model.get("features")
    if feats is None:
        feats = features

    X = df[feats].values if feats else df.values

    labels = None
    scores = None

    if hasattr(mdl, "predict"):
        labels = mdl.predict(X)
    if return_scores and hasattr(mdl, "decision_function"):
        scores = mdl.decision_function(X)
    elif return_scores and hasattr(mdl, "score_samples"):
        # Often higher = more normal; we invert in score() when returning pure scores
        scores = mdl.score_samples(X)

    if return_scores:
        return {"labels": labels, "scores": scores}
    return labels  # type: ignore[return-value]


def score(model: Any, df: pd.DataFrame) -> np.ndarray:
    """Return anomaly scores (higher = more anomalous), using whatever the model exposes.

    Accepts either a raw estimator or an artifact dict {"model": ..., "features": [...]}.
    """
    mdl = model.get("model", model) if isinstance(model, dict) else model
    feats = model.get("features") if isinstance(model, dict) else None
    X = df[feats].values if feats else df.values

    # Prefer decision_function (often higher = more normal for IF); invert to anomaly-high
    if hasattr(mdl, "decision_function"):
        return -np.asarray(mdl.decision_function(X), dtype=float)

    # score_samples: also invert so higher means more anomalous
    if hasattr(mdl, "score_samples"):
        return -np.asarray(mdl.score_samples(X), dtype=float)

    # Fallback: map predict {-1, 1} → {1.0, 0.0}
    if hasattr(mdl, "predict"):
        pred = mdl.predict(X)
        return (np.asarray(pred) == -1).astype(float)

    raise RuntimeError("No usable scoring method found on model")


# ---------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------

def load_current_model(registry_path: str | Path | None = None) -> tuple[Any, dict]:
    """Load the 'current_model' defined in a registry.json."""
    reg_path = Path(registry_path or "models/registry.json")
    if not reg_path.exists():
        raise FileNotFoundError(f"Registry file not found: {reg_path}")

    registry = json.loads(reg_path.read_text(encoding="utf-8"))
    current = registry.get("current_model")
    if not current:
        raise RuntimeError("No current_model entry in registry.json")

    model = load_model(current["path"])
    return model, current


__all__ = ["load_model", "load_current_model", "predict", "score"]
# Ensure the module is importable
if __name__ == "__main__":
    print("Log Guardian prediction utilities loaded successfully.")
# This allows the module to be run directly for testing purposes.
# It will not execute any code but confirms the module is importable.
# This is useful for debugging and ensuring the module structure is correct.
# You can add test cases or example usage here if needed.
"""Training utilities for Log Guardian anomaly detection models.

Provides functions to fit models (e.g., Isolation Forest), evaluate them,
and save artifacts + registry entries.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

LOGGER = logging.getLogger("log_guardian.modeling.train")


# ---------------------------------------------------------------------
# Train a model
# ---------------------------------------------------------------------

def train_isolation_forest(
    df: pd.DataFrame,
    features: List[str],
    *,
    n_estimators: int = 200,
    contamination: float | str = "auto",
    random_state: int = 42,
) -> IsolationForest:
    """Train an Isolation Forest model on selected features."""
    X = df[features].values
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    return model


# ---------------------------------------------------------------------
# Save model artifact
# ---------------------------------------------------------------------

def save_model(model: Any, features: List[str], out_dir: str | Path, version: str) -> Path:
    """Save trained model + metadata with joblib, return artifact path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    artifact_path = out_dir / f"isolation_forest_v{version}_{ts}.pkl"
    joblib.dump(
        {
            "model": model,
            "features": features,
            "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "version": version,
        },
        artifact_path,
    )
    LOGGER.info("Saved model to %s", artifact_path)
    return artifact_path


# ---------------------------------------------------------------------
# Update model registry
# ---------------------------------------------------------------------

def update_registry(
    registry_path: str | Path,
    model_name: str,
    version: str,
    artifact_path: Path,
    features: List[str],
) -> Dict[str, Any]:
    """Update the JSON registry with new model entry and return it."""
    registry_path = Path(registry_path)
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {
            "registry_name": "log_guardian_model_registry",
            "registry_type": "json",
            "registry_format": "v1",
            "registry_metadata": {
                "author": "Xcite",
                "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "version": "1.0",
            },
            "history": [],
        }

    entry = {
        "name": model_name,
        "version": version,
        "path": str(artifact_path),
        "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "features": features,
    }

    registry["current_model"] = entry
    registry.setdefault("history", []).append({k: entry[k] for k in ["name", "version", "path", "trained_at"]})

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Updated registry at %s", registry_path)
    return registry


# ---------------------------------------------------------------------
# High-level train entrypoint
# ---------------------------------------------------------------------

def train_and_register(
    df: pd.DataFrame,
    features: List[str],
    *,
    out_dir: str | Path = "models/artifacts",
    registry_path: str | Path = "models/registry.json",
    model_name: str = "isolation_forest",
    version: Optional[str] = None,
    n_estimators: int = 200,
    contamination: float | str = "auto",
    random_state: int = 42,
) -> Path:
    """Fit a model, save artifact, and update registry; return artifact path."""
    if version is None:
        # Simple timestamp version if not provided
        version = datetime.now().strftime("%Y.%m.%d.%H%M%S")

    model = train_isolation_forest(
        df,
        features,
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    artifact_path = save_model(model, features, out_dir, version)
    update_registry(registry_path, model_name, version, artifact_path, features)
    return artifact_path

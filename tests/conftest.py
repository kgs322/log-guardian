"""
Pytest configuration & shared fixtures for Log Guardian.

Also ensures modules under `src/` are importable inside tests by
inserting that directory at the front of sys.path, and provides a
back-compat alias so tests that import `readers` resolve to
`log_guardian.ingestion.readers`.

Provides:
- Temporary sample logs (auth + nginx)
- Feature DataFrames
- A tiny trained model artifact + registry.json
- FastAPI TestClient bound to the app using the temp registry
"""
from __future__ import annotations

# --- Make `src` importable for tests -----------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))  # highest priority

# --- Back-compat alias: some tests import `readers` at top-level --------------
# If your real module lives at log_guardian.ingestion.readers, alias it so
# `import readers` works.
try:
    import importlib
    if "readers" not in sys.modules:
        sys.modules["readers"] = importlib.import_module(
            "log_guardian.ingestion.readers"
        )
except Exception:
    # If the module truly doesn't exist yet, tests will still skip.
    pass

# -----------------------------------------------------------------------------
import json
import os
from typing import Dict, Iterator, Tuple  # noqa: F401

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

# Try to import the app factory and predict module (skip API fixtures if unavailable)
try:
    from log_guardian.api.app import create_app  # type: ignore
except Exception:  # pragma: no cover
    create_app = None  # type: ignore

try:
    from log_guardian.modeling import predict as predict_mod  # type: ignore
except Exception:  # pragma: no cover
    predict_mod = None  # type: ignore


# ---------------------------------------------------------------------------
# Sample log contents
# ---------------------------------------------------------------------------

AUTH_SAMPLE = """
Jan  7 12:01:02 web-1 sshd[1234]: Failed password for invalid user admin from 10.0.0.3 port 5555 ssh2
Jan  7 12:01:05 web-1 sshd[1234]: Failed password for root from 10.0.0.3 port 5556 ssh2
Jan  7 12:01:07 web-1 sshd[1234]: Accepted password for ubuntu from 10.0.0.4 port 6001 ssh2
""".strip()

NGINX_SAMPLE = """
127.0.0.1 - - [10/Oct/2020:13:55:36 +0000] "GET /index.html HTTP/1.1" 200 1024 "-" "curl/7.58.0"
127.0.0.1 - - [10/Oct/2020:13:55:40 +0000] "GET /does-not-exist HTTP/1.1" 404 0 "-" "curl/7.58.0"
127.0.0.1 - - [10/Oct/2020:13:55:45 +0000] "POST /login HTTP/1.1" 500 0 "-" "curl/7.58.0"
""".strip()


# ---------------------------------------------------------------------------
# Filesystem fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_project(tmp_path: Path) -> Dict[str, Path]:
    """Create a temporary project-like layout with data/ and models/."""
    d = {
        "root": tmp_path,
        "data": tmp_path / "data",
        "models": tmp_path / "models",
        "artifacts": tmp_path / "models" / "artifacts",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture()
def sample_logs(tmp_project: Dict[str, Path]) -> Dict[str, Path]:
    auth = tmp_project["data"] / "auth_small.log"
    nginx = tmp_project["data"] / "nginx_small.log"
    auth.write_text(AUTH_SAMPLE, encoding="utf-8")
    nginx.write_text(NGINX_SAMPLE, encoding="utf-8")
    return {"auth": auth, "nginx": nginx}


# ---------------------------------------------------------------------------
# Feature / model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_features() -> pd.DataFrame:
    """A minimal numeric feature matrix for model training/scoring tests."""
    return pd.DataFrame(
        {
            "failed_login_rate": [0.01, 0.40, 0.02, 0.35],
            "unique_ports": [5, 55, 6, 60],
            "reqs_per_min": [120, 900, 130, 850],
            "status_4xx_5xx_ratio": [0.03, 0.45, 0.02, 0.40],
        }
    )


@pytest.fixture()
def tiny_model_artifact(tmp_project: Dict[str, Path]) -> Tuple[Path, Dict[str, str]]:
    """Train a tiny IsolationForest and save a joblib artifact with feature list."""
    df = pd.DataFrame(
        {
            "failed_login_rate": [0.01, 0.40, 0.02, 0.35],
            "unique_ports": [5, 55, 6, 60],
            "reqs_per_min": [120, 900, 130, 850],
            "status_4xx_5xx_ratio": [0.03, 0.45, 0.02, 0.40],
        }
    )
    X = df.values
    model = IsolationForest(n_estimators=50, contamination="auto", random_state=0).fit(X)

    artifact = tmp_project["artifacts"] / "iforest_test.pkl"
    joblib.dump({"model": model, "features": list(df.columns)}, artifact)

    meta = {"name": "isolation_forest", "version": "test", "path": str(artifact)}
    return artifact, meta


@pytest.fixture()
def registry_with_current(tmp_project: Dict[str, Path], tiny_model_artifact) -> Path:
    """Write a registry.json pointing to the tiny model artifact."""
    artifact, meta = tiny_model_artifact
    reg = {
        "registry_name": "log_guardian_model_registry",
        "current_model": meta,
        "history": [meta],
    }
    reg_path = tmp_project["models"] / "registry.json"
    reg_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    return reg_path


# ---------------------------------------------------------------------------
# API client fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def api_client(registry_with_current: Path):
    if create_app is None:
        pytest.skip("FastAPI app not available")

    # Point the app to the temp registry
    os.environ["LOG_GUARDIAN_MODEL_REGISTRY"] = str(registry_with_current)

    from fastapi.testclient import TestClient

    app = create_app({})
    client = TestClient(app)
    try:
        yield client
    finally:
        # Cleanup env var
        os.environ.pop("LOG_GUARDIAN_MODEL_REGISTRY", None)


# ---------------------------------------------------------------------------
# Predict convenience
# ---------------------------------------------------------------------------

@pytest.fixture()
def score_with_tiny_model(registry_with_current: Path):
    """Return a function that scores a DataFrame using the tiny model via predict module."""
    if predict_mod is None:
        pytest.skip("predict module not available")

    def _scorer(df: pd.DataFrame) -> np.ndarray:
        # Load artifact
        info = json.loads(registry_with_current.read_text("utf-8")).get("current_model")
        obj = joblib.load(Path(info["path"]))
        model = obj.get("model", obj)
        feats = obj.get("features") or list(df.columns)
        X = df[feats].values

        # Prefer module-level scoring if present
        if hasattr(predict_mod, "score"):
            return predict_mod.score({"model": model, "features": feats}, df)
        if hasattr(model, "decision_function"):
            return -model.decision_function(X)
        if hasattr(model, "score_samples"):
            return -model.score_samples(X)
        if hasattr(model, "predict"):
            pred = model.predict(X)
            return (pred == -1).astype(float)
        raise RuntimeError("No scoring available")

    return _scorer

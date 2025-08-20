"""API tests for Log Guardian FastAPI app."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest


def test_healthz_ok(api_client):
    resp = api_client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert data.get("status") == "ok"


def test_score_returns_scores_and_flags(api_client, small_features, monkeypatch):
    """Ensure /score returns aligned scores and flags by monkeypatching predict.score."""
    try:
        import log_guardian.modeling.predict as p  # type: ignore
    except Exception:
        pytest.skip("predict module not available")

    def _score(model, df: pd.DataFrame):
        # Accept either a raw model or a dict artifact {model, features}
        obj = model
        if isinstance(obj, dict):
            mdl = obj.get("model", obj)
            feats: List[str] = obj.get("features") or list(df.columns)
        else:
            mdl = obj
            feats = list(df.columns)
        X = df[feats].values
        if hasattr(mdl, "decision_function"):
            return -mdl.decision_function(X)
        if hasattr(mdl, "score_samples"):
            return -mdl.score_samples(X)
        if hasattr(mdl, "predict"):
            pred = mdl.predict(X)
            return (pred == -1).astype(float)
        raise RuntimeError("No scoring method available")

    # Make /score use our patched scorer (works whether p.score exists or not)
    monkeypatch.setattr(p, "score", _score, raising=False)

    rows = small_features.to_dict(orient="records")
    resp = api_client.post("/score", json={"rows": rows})
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    assert "scores" in payload and "flagged" in payload
    assert isinstance(payload["scores"], list)
    assert isinstance(payload["flagged"], list)
    assert len(payload["scores"]) == len(rows)
    assert len(payload["flagged"]) == len(rows)
    assert set(payload["flagged"]).issubset({0, 1})
    for s in payload["scores"]:
        assert isinstance(s, (int, float))


def test_score_with_threshold(api_client, small_features, monkeypatch):
    """Test /score with deterministic scores and verify default flagging logic."""
    try:
        import log_guardian.modeling.predict as p  # type: ignore
    except Exception:
        pytest.skip("predict module not available")

    def _score(model, df: pd.DataFrame):
        # Deterministic scores from a known column; normalized to [0,1]
        col = "reqs_per_min"
        if col in df.columns:
            x = df[col].to_numpy(dtype=float)
            rng = np.ptp(x)  # NumPy 2.0+: function form, not x.ptp()
            if rng == 0:
                return np.zeros_like(x, dtype=float)
            return (x - x.min()) / rng
        return np.zeros(len(df), dtype=float)

    monkeypatch.setattr(p, "score", _score, raising=False)

    rows = small_features.to_dict(orient="records")
    resp = api_client.post("/score", json={"rows": rows})
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    assert "scores" in payload and "flagged" in payload
    scores = np.asarray(payload["scores"], dtype=float)
    flags = np.asarray(payload["flagged"], dtype=int)

    # Emulate the API's default flagging: mean + 2*std
    cutoff = scores.mean() + 2 * scores.std()
    expected_flags = (scores >= cutoff).astype(int)

    assert flags.tolist() == expected_flags.tolist()
    
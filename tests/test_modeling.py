import pandas as pd
import pytest

from log_guardian.modeling.train import train_isolation_forest, train_and_register
from log_guardian.modeling.postprocess import attach_flags


def make_dummy_df():
    return pd.DataFrame({
        "failed_login_rate": [0.1, 0.5, 0.0, 0.3, 0.7],
        "unique_ports": [2, 10, 1, 5, 7],
        "reqs_per_min": [30, 200, 10, 50, 100],
        "status_4xx_5xx_ratio": [0.01, 0.2, 0.0, 0.05, 0.3],
    })


def test_train_isolation_forest_runs(tmp_path):
    df = make_dummy_df()
    features = df.columns.tolist()
    model = train_isolation_forest(df, features)
    assert hasattr(model, "predict")
    preds = model.predict(df[features])
    assert len(preds) == len(df)


def test_train_and_register_creates_artifact(tmp_path):
    df = make_dummy_df()
    features = df.columns.tolist()
    out_dir = tmp_path / "models"
    reg_path = tmp_path / "registry.json"

    artifact = train_and_register(
        df, features, out_dir=out_dir, registry_path=reg_path
    )

    assert artifact.exists()
    assert reg_path.exists()
    content = reg_path.read_text()
    assert "isolation_forest" in content


def test_attach_flags_adds_column():
    df = make_dummy_df()
    flags = [1, -1, 1, -1, 1]
    out = attach_flags(df, flags, col="flagged")
    assert "flagged" in out.columns
    assert list(out["flagged"]) == flags
def test_attach_flags_handles_empty():
    df = pd.DataFrame(columns=["col1", "col2"])
    out = attach_flags(df, [], col="flagged")
    assert "flagged" in out.columns
    assert out["flagged"].empty
    assert out.shape == (0, 3)  # original + flagged column
def test_attach_flags_with_existing_column():
    df = make_dummy_df()
    df["flagged"] = [0, 0, 0, 0, 0]
    flags = [1, -1, 1, -1, 1]
    out = attach_flags(df, flags, col="flagged")
    assert "flagged" in out.columns
    assert list(out["flagged"]) == flags
    assert len(out) == len(df)
    
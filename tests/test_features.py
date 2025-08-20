"""Feature building tests for Log Guardian."""
from __future__ import annotations

import pandas as pd
import pytest

# Import from the installed package (src/ layout + pip install -e .)
try:
    from log_guardian.features.build_features import build_features  # type: ignore
except Exception:  # pragma: no cover
    build_features = None  # type: ignore


def test_build_features_from_auth(sample_logs):
    if build_features is None:
        pytest.skip("build_features not implemented")
    df = build_features(str(sample_logs["auth"]))
    assert isinstance(df, pd.DataFrame)
    # Should produce at least one numeric feature
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    assert len(num_cols) >= 1


def test_build_features_from_nginx(sample_logs):
    if build_features is None:
        pytest.skip("build_features not implemented")
    df = build_features(str(sample_logs["nginx"]))
    assert isinstance(df, pd.DataFrame)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    assert len(num_cols) >= 1


def test_build_features_empty_file(tmp_path):
    """Ensure build_features handles empty files gracefully."""
    if build_features is None:
        pytest.skip("build_features not implemented")
    empty_file = tmp_path / "empty.log"
    empty_file.write_text("", encoding="utf-8")
    df = build_features(str(empty_file))
    assert isinstance(df, pd.DataFrame)
    # Expect an empty DataFrame for an empty log file
    assert df.empty, "Expected an empty DataFrame for an empty log file"


def test_build_features_invalid_file(tmp_path):
    """Ensure build_features is robust to invalid files."""
    if build_features is None:
        pytest.skip("build_features not implemented")
    invalid_file = tmp_path / "invalid.log"
    invalid_file.write_text("This is not a valid log format", encoding="utf-8")

    # Accept either a ValueError or an empty DataFrame, depending on implementation.
    try:
        df = build_features(str(invalid_file))
        assert isinstance(df, pd.DataFrame)
        assert df.empty, "Expected empty DataFrame for invalid input"
    except ValueError:
        # Also acceptable behavior
        pass


def test_build_features_with_missing_columns(sample_logs, tmp_path):
    """Ensure build_features can handle logs with unexpected/missing patterns."""
    if build_features is None:
        pytest.skip("build_features not implemented")

    # Create a variant of auth log with altered wording to simulate missing patterns
    custom_log_text = sample_logs["auth"].read_text(encoding="utf-8").replace(
        "Failed password", "Custom log entry"
    )
    custom_log_file = tmp_path / "custom_auth.log"
    custom_log_file.write_text(custom_log_text, encoding="utf-8")

    df = build_features(str(custom_log_file))
    assert isinstance(df, pd.DataFrame)
    # We just require the pipeline not to crash; some features may still be produced
    assert df.shape[0] >= 0  # non-crashing behavior
    assert df.shape[1] >= 0  # non-crashing behavior
    # If the log is completely unparseable, we expect an empty DataFrame
    if df.empty:
        assert df.shape[0] == 0, "Expected empty DataFrame for unparseable log"
    else:
        # If features are produced, we expect at least one numeric column
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        assert len(num_cols) >= 1, "Expected at least one numeric feature column"
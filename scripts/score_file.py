#!/usr/bin/env python3
"""
Score a file with the latest Log Guardian model.

Examples
--------
# Raw log → features → scores → CSV
python scripts/score_file.py \
  --input data/samples/auth_small.log \
  --output data/processed/auth_small_scored.csv

# Already-built features (CSV) → scores → Parquet
python scripts/score_file.py \
  --input data/processed/features_auth.csv \
  --output data/processed/features_auth_scored.parquet \
  --features

# Override threshold & emit anomaly flags and top-K
python scripts/score_file.py \
  --input data/samples/nginx_small.log \
  --output data/processed/nginx_scored.jsonl \
  --format jsonl --topk 50 --threshold 0.75

Design notes
------------
- Loads the latest model from models/registry.json unless --model is provided.
- If input looks like a raw log (non .csv/.parquet/.jsonl) it will be parsed via build_features().
- If input is a dataframe file (csv/parquet/jsonl) it will be used as-is unless --raw is passed.
- Produces anomaly_score and anomaly_flag using threshold/percentile.
- Writes CSV, Parquet, or JSONL based on --format or output extension.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import joblib

# Internal modules (soft imports with friendly errors)
try:
    from src.log_guardian.features.build_features import build_features
except Exception as e:  # pragma: no cover
    build_features = None  # type: ignore

try:
    from src.log_guardian.config import load_config
except Exception:
    load_config = None  # type: ignore

try:
    from src.log_guardian.modeling import predict
except Exception:
    predict = None  # type: ignore


# -----------------------------
# Logging
# -----------------------------
LOGGER = logging.getLogger("log_guardian.scoring")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# -----------------------------
# Registry / Model loading
# -----------------------------

def load_latest_model(
    registry_path: Path = Path("models/registry.json"),
    override_model_path: Optional[Path] = None,
) -> Tuple[object, dict]:
    """Load a model either from an explicit path or the registry's current_model.

    Returns (model_object, model_info_dict)
    """
    if override_model_path is not None:
        if not override_model_path.exists():
            raise FileNotFoundError(f"Model file not found: {override_model_path}")
        model = joblib.load(override_model_path)
        model_info = {"name": Path(override_model_path).stem, "version": "(override)", "path": str(override_model_path)}
        return model, model_info

    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    if "current_model" not in registry:
        raise KeyError("Registry missing 'current_model'")

    model_info = registry["current_model"]
    model_path = Path(model_info["path"])  # expected relative to repo root

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    return model, model_info


# -----------------------------
# IO helpers
# -----------------------------

def infer_is_features_file(path: Path) -> bool:
    """Heuristic: CSV/Parquet/JSONL are treated as already-built features."""
    return path.suffix.lower() in {".csv", ".parquet", ".jsonl"}


def read_dataframe(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".jsonl" or suf == ".json":
        # Expect one JSON object per line
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported dataframe file: {path}")


def write_dataframe(df: pd.DataFrame, out_path: Path, fmt: Optional[str] = None) -> None:
    fmt_final = (fmt or out_path.suffix.lower().lstrip(".") or "csv").lower()
    if fmt_final == "csv":
        df.to_csv(out_path, index=False)
    elif fmt_final == "parquet":
        df.to_parquet(out_path, index=False)
    elif fmt_final == "jsonl":
        df.to_json(out_path, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported output format: {fmt_final}")


# -----------------------------
# Scoring helpers
# -----------------------------

def compute_anomaly_flag(
    scores: pd.Series | np.ndarray,
    threshold: Optional[float] = None,
    percentile: Optional[float] = None,
) -> pd.Series:
    """Return 1 if score >= threshold (or >= percentile-cut), else 0.

    Priority: explicit threshold > percentile > no flagging (all zeros).
    """
    s = pd.Series(scores).astype(float)

    if threshold is not None:
        flags = (s >= float(threshold)).astype(int)
        return flags

    if percentile is not None:
        if not (0 < percentile < 100):
            raise ValueError("percentile must be in (0, 100)")
        cut = np.percentile(s, percentile)
        flags = (s >= cut).astype(int)
        return flags

    # Default: no flagging
    return pd.Series(np.zeros(len(s), dtype=int))


# -----------------------------
# Main
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score a log or features file with the latest model")
    p.add_argument("--input", required=True, help="Path to raw log OR features file")
    p.add_argument("--output", required=True, help="Path to save scored results (csv/parquet/jsonl)")
    p.add_argument("--config", default="configs/prod.yaml", help="Config YAML (for defaults like threshold)")
    p.add_argument("--registry", default="models/registry.json", help="Model registry path")
    p.add_argument("--model", default=None, help="Override: direct path to a .pkl/.joblib model file")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--raw", action="store_true", help="Force treat input as raw log")
    mode.add_argument("--features", action="store_true", help="Force treat input as features dataframe")

    p.add_argument("--threshold", type=float, default=None, help="Anomaly threshold (overrides config)")
    p.add_argument("--percentile", type=float, default=None, help="Flag anomalies at this percentile cutoff (0-100)")
    p.add_argument("--format", choices=["csv", "parquet", "jsonl"], default=None, help="Output format (else inferred from extension)")
    p.add_argument("--topk", type=int, default=None, help="If set, keep only the top-K highest anomaly scores")

    p.add_argument("--id-col", default=None, help="Optional ID column to keep first when sorting by score")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    registry_path = Path(args.registry)
    override_model = Path(args.model) if args.model else None

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    # Load config if available
    cfg = {}
    if load_config is not None and Path(args.config).exists():
        try:
            cfg = load_config(args.config) or {}
            LOGGER.info("Loaded config: %s", args.config)
        except Exception as e:  # pragma: no cover
            LOGGER.warning("Failed to load config %s: %s", args.config, e)

    # Determine input mode
    if args.raw:
        is_features = False
    elif args.features:
        is_features = True
    else:
        is_features = infer_is_features_file(in_path)

    # Build or read features
    if is_features:
        LOGGER.info("Reading features from %s", in_path)
        features_df = read_dataframe(in_path)
    else:
        if build_features is None:
            raise ImportError("build_features is not available. Ensure src/log_guardian/features/build_features.py exists.")
        LOGGER.info("Parsing raw log via build_features(): %s", in_path)
        features_df = build_features(str(in_path))

    if features_df is None or len(features_df) == 0:
        LOGGER.warning("No features were generated/read; exiting with empty output.")
        empty = features_df if isinstance(features_df, pd.DataFrame) else pd.DataFrame()
        write_dataframe(empty, out_path, fmt=args.format)
        return

    # Load model
    model, model_info = load_latest_model(registry_path=registry_path, override_model_path=override_model)
    LOGGER.info("Loaded model: %s v%s (%s)", model_info.get("name"), model_info.get("version"), model_info.get("path"))

    # Score
    if predict is None or not hasattr(predict, "score"):
        raise ImportError("predict.score is not available. Implement score(model, features_df) in src/log_guardian/modeling/predict.py")

    scores = predict.score(model, features_df)
    scores = pd.Series(scores, name="anomaly_score").astype(float)

    # Threshold / percentile logic with config fallbacks
    cfg_threshold = None
    cfg_percentile = None
    try:
        # Prefer scoring.threshold, then model.threshold
        cfg_threshold = (
            cfg.get("scoring", {}).get("threshold")
            if isinstance(cfg, dict)
            else None
        )
        if cfg_threshold is None:
            cfg_threshold = cfg.get("model", {}).get("threshold") if isinstance(cfg, dict) else None
        cfg_percentile = cfg.get("scoring", {}).get("percentile") if isinstance(cfg, dict) else None
    except Exception:
        pass

    threshold = args.threshold if args.threshold is not None else cfg_threshold
    percentile = args.percentile if args.percentile is not None else cfg_percentile

    flags = compute_anomaly_flag(scores, threshold=threshold, percentile=percentile)

    # Assemble results
    results = features_df.copy()
    results.insert(len(results.columns), "anomaly_score", scores.values)
    results.insert(len(results.columns), "anomaly_flag", flags.values)

    # If requested, keep only top-K rows by score
    if args.topk is not None and args.topk > 0:
        sort_cols = ["anomaly_score"]
        if args.id_col and args.id_col in results.columns:
            sort_cols = ["anomaly_score", args.id_col]
        results = results.sort_values(by="anomaly_score", ascending=False).head(args.topk)

    # Attach minimal metadata as attributes for certain formats
    results.attrs["model_name"] = model_info.get("name")
    results.attrs["model_version"] = model_info.get("version")
    results.attrs["model_path"] = model_info.get("path")

    # Write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_dataframe(results, out_path, fmt=args.format)

    # Friendly stdout summary
    total = len(results)
    flagged = int(results["anomaly_flag"].sum()) if "anomaly_flag" in results else 0
    LOGGER.info("Saved scored output → %s", out_path)
    LOGGER.info("Rows: %d | Flagged: %d | Threshold: %s | Percentile: %s", total, flagged, threshold, percentile)
    print(json.dumps({
        "output": str(out_path),
        "rows": total,
        "flagged": flagged,
        "model": model_info,
        "threshold": threshold,
        "percentile": percentile,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
# This script is used to score a log file using the latest trained model.
# It can handle both raw logs and pre-built feature files, applying anomaly detection   
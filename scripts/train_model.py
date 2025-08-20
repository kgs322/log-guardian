#!/usr/bin/env python3
"""
Train a Log Guardian model and update the registry.

Examples
--------
# Train from raw logs, write artifact + update registry
python scripts/train_model.py \
  --name isolation_forest \
  --input data/samples/auth_small.log \
  --output-dir models/artifacts \
  --registry models/registry.json \
  --config configs/dev.yaml

# Train from a pre-built features dataframe
python scripts/train_model.py \
  --name isolation_forest \
  --features data/processed/features_auth.csv \
  --version 0.2.0 \
  --notes "added status_4xx_5xx_ratio"

Design
------
- Delegates actual model training to `src.log_guardian.modeling.train.train_model(features_df, cfg)`
  which should return `(fitted_model, metrics: dict)`.
- Persists joblib artifact to `--output-dir` with a name derived from `--name` and `--version`.
- Updates/creates `models/registry.json` with `current_model` and appends to `history`.
- Captures feature columns used and timestamp metadata.
- Accepts either raw logs via `--input` (then calls build_features) or a features file via `--features`.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone

import joblib
import pandas as pd

# Internal modules (soft imports w/ friendly errors)
try:
    from src.log_guardian.features.build_features import build_features
except Exception:
    build_features = None  # type: ignore

try:
    from src.log_guardian.config import load_config
except Exception:
    load_config = None  # type: ignore

try:
    from src.log_guardian.modeling.train import train_model as modeling_train
except Exception:
    modeling_train = None  # type: ignore

LOGGER = logging.getLogger("log_guardian.train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# -----------------------------
# IO helpers
# -----------------------------

def read_dataframe(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf in {".json", ".jsonl"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported features file: {path}")


# -----------------------------
# Registry helpers
# -----------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_registry(path: Path) -> Dict:
    if not path.exists():
        # bootstrap structure
        return {
            "registry_name": "log_guardian_model_registry",
            "registry_type": "json",
            "registry_format": "v1",
            "registry_metadata": {
                "author": "unknown",
                "created_at": _now_iso(),
                "version": "1.0",
            },
            "current_model": None,
            "history": [],
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(path: Path, registry: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    tmp.replace(path)


def _bump_version(prev: Optional[str], bump: str = "minor") -> str:
    if not prev:
        return "0.1.0"
    parts = prev.split(".")
    parts = [int(p) for p in parts] + [0] * (3 - len(parts))
    major, minor, patch = parts[:3]
    if bump == "major":
        major, minor, patch = major + 1, 0, 0
    elif bump == "patch":
        patch += 1
    else:
        minor += 1
    return f"{major}.{minor}.{patch}"


def _artifact_name(model_name: str, version: str, ext: str = ".pkl") -> str:
    safe = model_name.lower().replace(" ", "_")
    return f"{safe}_v{version}{ext}"


# -----------------------------
# Main training flow
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Log Guardian model and update registry")
    p.add_argument("--name", required=True, help="Model family/name, e.g., isolation_forest")

    src_grp = p.add_mutually_exclusive_group(required=True)
    src_grp.add_argument("--input", help="Path to raw log file to parse into features")
    src_grp.add_argument("--features", help="Path to pre-built features (csv/parquet/jsonl)")

    p.add_argument("--config", default=None, help="YAML config with training hyperparams")
    p.add_argument("--output-dir", default="models/artifacts", help="Directory to write the model artifact")
    p.add_argument("--registry", default="models/registry.json", help="Model registry path to update")

    # Versioning
    p.add_argument("--version", default=None, help="Explicit semantic version (e.g., 0.2.0)")
    p.add_argument("--bump", choices=["major","minor","patch"], default="minor", help="If --version absent, bump relative to current")

    # Metadata
    p.add_argument("--notes", default=None, help="Free-form notes about this training run")
    p.add_argument("--author", default=None, help="Override author in registry metadata")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    # Load/merge config
    cfg: Dict = {}
    if load_config and args.config and Path(args.config).exists():
        try:
            cfg = load_config(args.config) or {}
            LOGGER.info("Loaded config: %s", args.config)
        except Exception as e:
            LOGGER.warning("Failed to load config %s: %s", args.config, e)

    # Features in
    if args.features:
        feat_path = Path(args.features)
        if not feat_path.exists():
            raise FileNotFoundError(f"Features file not found: {feat_path}")
        features_df = read_dataframe(feat_path)
        LOGGER.info("Loaded features: %s (rows=%d, cols=%d)", feat_path, len(features_df), features_df.shape[1])
    else:
        if build_features is None:
            raise ImportError("build_features is not available. Ensure src/log_guardian/features/build_features.py exists.")
        log_path = Path(args.input)
        if not log_path.exists():
            raise FileNotFoundError(f"Input log not found: {log_path}")
        LOGGER.info("Building features from raw: %s", log_path)
        features_df = build_features(str(log_path))
        LOGGER.info("Built features (rows=%d, cols=%d)", len(features_df), features_df.shape[1])

    if features_df is None or len(features_df) == 0:
        raise RuntimeError("No features available for training")

    # Train model using modeling.train API
    if modeling_train is None:
        raise ImportError("modeling.train.train_model is not available. Implement train_model(features_df, cfg) in src/log_guardian/modeling/train.py")

    LOGGER.info("Training model '%s'…", args.name)
    model, metrics = modeling_train(features_df, cfg)  # type: ignore

    # Determine version and artifact path
    registry_path = Path(args.registry)
    registry = _load_registry(registry_path)

    prev_version = None
    if isinstance(registry.get("current_model"), dict) and registry["current_model"].get("name") == args.name:
        prev_version = registry["current_model"].get("version")

    version = args.version or _bump_version(prev_version, bump=args.bump)
    artifact_dir = Path(args.output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / _artifact_name(args.name, version)

    # Persist artifact
    joblib.dump(model, artifact_path)
    LOGGER.info("Saved model artifact → %s", artifact_path)

    # Collect feature list for registry
    try:
        feature_list = list(features_df.columns)
    except Exception:
        feature_list = []

    # Update registry
    entry = {
        "name": args.name,
        "version": version,
        "path": str(artifact_path.as_posix()),
        "trained_at": _now_iso(),
        "features": feature_list,
        "metrics": metrics or {},
        "notes": args.notes or "",
    }

    # history append (keep previous current)
    if registry.get("current_model"):
        registry.setdefault("history", []).append(registry["current_model"])  # type: ignore[arg-type]

    registry["current_model"] = entry

    if args.author:
        registry.setdefault("registry_metadata", {})["author"] = args.author

    _save_registry(registry_path, registry)
    LOGGER.info("Updated registry: %s", registry_path)

    # Friendly JSON print for tooling
    print(json.dumps({
        "artifact": str(artifact_path),
        "version": version,
        "metrics": metrics,
        "registry": str(registry_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
# This script is used to train a Log Guardian model and update the model registry.
# It can handle both raw logs and pre-built feature files, applying anomaly detection.
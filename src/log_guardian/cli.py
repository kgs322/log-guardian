#!/usr/bin/env python3
"""Log Guardian CLI

Unified command-line interface for training, scoring, and serving the API.

Examples
--------
# Train from raw logs and update registry
python -m src.log_guardian.cli train --input data/raw

# Score a file and save CSV
python -m src.log_guardian.cli score --input data/samples/auth_small.log --output out.csv

# Serve FastAPI (dev reload)
python -m src.log_guardian.cli serve --reload
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

# Internal imports (kept defensive so CLI still works during scaffolding)
try:
    from .features import build_features  # type: ignore
except Exception:
    build_features = None  # type: ignore

try:
    from .modeling import predict as predict_mod  # type: ignore
except Exception:
    predict_mod = None  # type: ignore

try:
    from .modeling.train import train_and_register  # type: ignore
except Exception:
    train_and_register = None  # type: ignore

LOGGER = logging.getLogger("log_guardian.cli")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_model_from_registry(registry_path: Path, override_model: Optional[Path] = None) -> tuple[Any, dict]:
    if override_model is not None:
        if not override_model.exists():
            raise FileNotFoundError(f"Model file not found: {override_model}")
        model = joblib.load(override_model)
        return model, {"name": override_model.stem, "version": "(override)", "path": str(override_model)}

    reg = _load_registry(registry_path)
    info = reg.get("current_model")
    if not info:
        raise RuntimeError("No current_model in registry.json")
    model_path = Path(info["path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    return model, info


def _read_df(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf in {".jsonl", ".json"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported dataframe file: {path}")


def _is_features_file(path: Path) -> bool:
    return path.suffix.lower() in {".csv", ".parquet", ".jsonl"}


def _score_with_model(model: Any, df: pd.DataFrame):
    """Return a 1D array-like of anomaly scores using whichever interface is available."""
    # Preferred: predict.score(model, df)
    if predict_mod is not None and hasattr(predict_mod, "score"):
        return predict_mod.score(model, df)

    # Fallback to our generic predict API if present
    if predict_mod is not None and hasattr(predict_mod, "predict"):
        out = predict_mod.predict(df, model=model, return_scores=True)
        scores = out.get("scores") if isinstance(out, dict) else None
        if scores is not None:
            return scores

    # Direct model methods
    if hasattr(model, "decision_function"):
        # Higher = more normal in some impls; we invert to be anomaly-high
        return -model.decision_function(df.values)
    if hasattr(model, "score_samples"):
        return -model.score_samples(df.values)
    if hasattr(model, "predict"):
        # Map -1 (anomaly) to 1.0, 1 (normal) to 0.0
        pred = model.predict(df.values)
        return (pred == -1).astype(float)

    raise RuntimeError("No usable scoring method found on model/predict module")


# ---------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    if train_and_register is None:
        raise ImportError("modeling.train.train_and_register not available")

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    if args.features or _is_features_file(in_path):
        df = _read_df(in_path)
    else:
        if build_features is None:
            raise ImportError("features.build_features not available; cannot parse raw logs")
        # Build features from raw
        if in_path.is_file():
            df = build_features(str(in_path))
        else:
            # Directory: simple gather of all files
            frames = []
            for p in in_path.rglob("*"):
                if p.is_file() and not _is_features_file(p):
                    frames.append(build_features(str(p)))
            if not frames:
                raise RuntimeError("No logs found to build features from")
            df = pd.concat(frames, ignore_index=True)

    feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in {"label"}]
    artifact = train_and_register(
        df,
        feature_cols,
        out_dir=args.out_dir,
        registry_path=args.registry,
        model_name=args.model_name,
        version=args.version,
    )
    print(json.dumps({"artifact": str(artifact), "n_features": len(feature_cols), "rows": len(df)}, ensure_ascii=False))


def cmd_score(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    # Build/read features
    if args.features or _is_features_file(in_path):
        df = _read_df(in_path)
    else:
        if build_features is None:
            raise ImportError("features.build_features not available; cannot parse raw logs")
        df = build_features(str(in_path))

    # Load model
    model, info = _load_model_from_registry(Path(args.registry), Path(args.model) if args.model else None)

    # Score
    scores = _score_with_model(model, df)
    scores = pd.Series(scores, name="anomaly_score").astype(float)

    # Flags (optional simple scheme)
    flags = (scores >= scores.mean() + 2 * scores.std()).astype(int)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = df.copy()
    results["anomaly_score"] = scores.values
    results["anomaly_flag"] = flags.values

    suf = (args.format or out_path.suffix.lstrip(".") or "csv").lower()
    if suf == "csv":
        results.to_csv(out_path, index=False)
    elif suf == "parquet":
        results.to_parquet(out_path, index=False)
    elif suf == "jsonl":
        results.to_json(out_path, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported output format: {suf}")

    print(json.dumps({
        "output": str(out_path),
        "rows": len(results),
        "flagged": int(results["anomaly_flag"].sum()),
        "model": info,
    }, ensure_ascii=False))


def cmd_serve(args: argparse.Namespace) -> None:
    # Re-use scripts/serve_api.py behavior via environment variables and uvicorn
    import os
    import uvicorn
    from .api.app import create_app  # type: ignore

    # Export basic env for the app layer
    if args.registry:
        os.environ["LOG_GUARDIAN_MODEL_REGISTRY"] = str(Path(args.registry).resolve())
    if args.model:
        os.environ["LOG_GUARDIAN_MODEL_PATH"] = str(Path(args.model).resolve())

    app = create_app({})
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
    )


# ---------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="log-guardian", description="Log Guardian CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    pt = sub.add_parser("train", help="Train a model and update registry")
    pt.add_argument("--input", required=True, help="Path to raw logs dir/file or features file")
    pt.add_argument("--features", action="store_true", help="Treat input as features file")
    pt.add_argument("--out-dir", default="models/artifacts")
    pt.add_argument("--registry", default="models/registry.json")
    pt.add_argument("--model-name", default="isolation_forest")
    pt.add_argument("--version", default=None)
    pt.set_defaults(func=cmd_train)

    # score
    ps = sub.add_parser("score", help="Score a log/features file")
    ps.add_argument("--input", required=True)
    ps.add_argument("--output", required=True)
    ps.add_argument("--features", action="store_true", help="Treat input as features file")
    ps.add_argument("--registry", default="models/registry.json")
    ps.add_argument("--model", default=None, help="Override model path")
    ps.add_argument("--format", choices=["csv", "parquet", "jsonl"], default=None)
    ps.set_defaults(func=cmd_score)

    # serve
    pv = sub.add_parser("serve", help="Serve FastAPI app")
    pv.add_argument("--host", default="0.0.0.0")
    pv.add_argument("--port", type=int, default=8000)
    pv.add_argument("--log-level", default="info")
    pv.add_argument("--reload", action="store_true")
    pv.add_argument("--registry", default="models/registry.json")
    pv.add_argument("--model", default=None)
    pv.set_defaults(func=cmd_serve)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
# ---------------------------------------------------------------------
# This is the main entrypoint for the CLI
# ---------------------------------------------------------------------
# It provides a unified interface for training, scoring, and serving the Log Guardian API.
# It can be run directly or imported as a module.
# ---------------------------------------------------------------------
# This file is part of the Log Guardian project.
# It is licensed under the Apache License 2.0.
# See LICENSE for details.
# ---------------------------------------------------------------------
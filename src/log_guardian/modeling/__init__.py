"""Modeling package: training, prediction, and post-processing utilities."""
from __future__ import annotations

# Training / registry
from .train import (
    train_isolation_forest,
    save_model,
    update_registry,
    train_and_register,
)

# Prediction
from .predict import (
    load_model,
    load_current_model,
    predict,
    score,
)

# Post-processing
from .postprocess import (
    apply_threshold,
    topk_indices,
    attach_scores,
    attach_flags,
    keep_topk,
    ZScoreCalibrator,
)

__all__ = [
    # train / registry
    "train_isolation_forest",
    "save_model",
    "update_registry",
    "train_and_register",
    # predict
    "load_model",
    "load_current_model",
    "predict",
    "score",
    # postprocess
    "apply_threshold",
    "topk_indices",
    "attach_scores",
    "attach_flags",
    "keep_topk",
    "ZScoreCalibrator",
]

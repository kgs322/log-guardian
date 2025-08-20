"""Post-processing utilities for Log Guardian modeling.

These helpers turn raw anomaly scores into actionable outputs: flags,
percentile cuts, and top-K selections. They are used by CLI and API layers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Basic thresholding / flagging
# ---------------------------------------------------------------------

def apply_threshold(
    scores: Sequence[float] | np.ndarray | pd.Series,
    *,
    threshold: float | None = None,
    percentile: float | None = None,
) -> Tuple[pd.Series, float | None]:
    """Compute binary flags from scores.

    Priority: explicit ``threshold`` > ``percentile`` > no flags (all zeros).

    Returns
    -------
    flags : pd.Series of {0,1}
    cutoff : float or None
        The numeric cutoff used (threshold value or percentile-derived), else None.
    """
    s = pd.Series(scores, dtype=float).copy()

    if threshold is not None:
        flags = (s >= float(threshold)).astype(int)
        return flags, float(threshold)

    if percentile is not None:
        if not (0 < percentile < 100):
            raise ValueError("percentile must be in (0, 100)")
        cut = float(np.percentile(s.values, percentile))
        flags = (s >= cut).astype(int)
        return flags, cut

    # Default: no flagging
    return pd.Series(np.zeros(len(s), dtype=int)), None


# ---------------------------------------------------------------------
# Top-K utilities
# ---------------------------------------------------------------------

def topk_indices(scores: Sequence[float] | np.ndarray | pd.Series, k: int) -> List[int]:
    """Return indices of the top-K highest scores (stable order by score desc).

    If k >= n, returns range(n).
    """
    s = np.asarray(scores, dtype=float)
    n = s.shape[0]
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))
    # argpartition is O(n); tie-break with argsort of the partitioned slice
    idx = np.argpartition(-s, k - 1)[:k]
    # Stable sort the selected indices by score desc
    idx_sorted = idx[np.argsort(-s[idx], kind="stable")]
    return idx_sorted.tolist()


# ---------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------

def attach_scores(
    df: pd.DataFrame,
    scores: Sequence[float] | np.ndarray | pd.Series,
    *,
    col: str = "anomaly_score",
) -> pd.DataFrame:
    """Return a copy of df with a score column appended (float)."""
    out = df.copy()
    out[col] = pd.Series(scores, dtype=float).values
    return out


def attach_flags(
    df: pd.DataFrame,
    flags: Sequence[int] | np.ndarray | pd.Series,
    *,
    col: str = "anomaly_flag",
) -> pd.DataFrame:
    """Return a copy of df with a flag column appended (int)."""
    out = df.copy()
    out[col] = pd.Series(flags, dtype=int).values
    return out


def keep_topk(df: pd.DataFrame, k: int, *, score_col: str = "anomaly_score") -> pd.DataFrame:
    """Return only the top-K rows by score (descending)."""
    if k is None or k <= 0:
        return df
    return df.sort_values(by=score_col, ascending=False).head(k)


# ---------------------------------------------------------------------
# Simple calibration helpers (optional)
# ---------------------------------------------------------------------
@dataclass
class ZScoreCalibrator:
    """Z-score normalization for scores (mean/std from a reference window)."""
    mean_: float
    std_: float

    @classmethod
    def fit(cls, scores: Sequence[float] | np.ndarray | pd.Series) -> "ZScoreCalibrator":
        s = pd.Series(scores, dtype=float)
        return cls(mean_=float(s.mean()), std_=float(s.std(ddof=0) or 1.0))

    def transform(self, scores: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        return (s - self.mean_) / (self.std_ if self.std_ != 0 else 1.0)

    def fit_transform(self, scores: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
        calib = self.fit(scores)
        return calib.transform(scores)

"""Heuristic rule signatures for Log Guardian.

These rules complement ML scoring by flagging clear-cut patterns
(e.g., brute-force attempts, HTTP error storms, port fan-out spikes).

Each rule receives a **features DataFrame** and returns a boolean mask
aligned to the DataFrame index. `apply_all` aggregates results and renders
an alerts DataFrame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _col(df: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    """Return a column if present; else a Series of default values."""
    return df[name] if name in df.columns else pd.Series([default] * len(df), index=df.index)


@dataclass(frozen=True)
class Rule:
    name: str
    description: str
    fn: Callable[[pd.DataFrame, Dict[str, float]], pd.Series]
    defaults: Dict[str, float]

    def run(self, df: pd.DataFrame, overrides: Optional[Dict[str, float]] = None) -> pd.Series:
        cfg = {**self.defaults, **(overrides or {})}
        mask = self.fn(df, cfg).astype(bool)
        return mask.reindex(df.index, fill_value=False)


# ---------------------------------------------------------------------
# Rule implementations (vectorized)
# ---------------------------------------------------------------------

def _rule_bruteforce(df: pd.DataFrame, cfg: Dict[str, float]) -> pd.Series:
    """High failed login rate and elevated req rate → likely brute force."""
    failed = _col(df, "failed_login_rate", 0.0).astype(float)
    rpm = _col(df, "reqs_per_min", 0.0).astype(float)
    return (failed >= cfg["min_failed_login_rate"]) & (rpm >= cfg["min_reqs_per_min"])  # type: ignore


def _rule_http_error_storm(df: pd.DataFrame, cfg: Dict[str, float]) -> pd.Series:
    """Sustained high 4xx/5xx ratio indicates an error storm / attack surface scan."""
    ratio = _col(df, "status_4xx_5xx_ratio", 0.0).astype(float)
    rpm = _col(df, "reqs_per_min", 0.0).astype(float)
    return (ratio >= cfg["min_error_ratio"]) & (rpm >= cfg["min_reqs_per_min"])  # type: ignore


def _rule_port_fanout(df: pd.DataFrame, cfg: Dict[str, float]) -> pd.Series:
    """Spike in unique destination ports suggests scanning/propagation."""
    ports = _col(df, "unique_ports", 0.0).astype(float)
    return ports >= cfg["min_unique_ports"]  # type: ignore


def _rule_request_spike(df: pd.DataFrame, cfg: Dict[str, float]) -> pd.Series:
    """Generic request flood based on z-score of reqs_per_min."""
    rpm = _col(df, "reqs_per_min", 0.0).astype(float)
    mean = float(rpm.mean())
    std = float(rpm.std(ddof=0) or 1.0)
    z = (rpm - mean) / std
    return z >= cfg["min_rpm_z"]  # type: ignore


# ---------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------

RULES: List[Rule] = [
    Rule(
        name="bruteforce",
        description="High failed-login rate with elevated traffic",
        fn=_rule_bruteforce,
        defaults={"min_failed_login_rate": 0.2, "min_reqs_per_min": 200.0},
    ),
    Rule(
        name="http_error_storm",
        description="High 4xx/5xx ratio with traffic",
        fn=_rule_http_error_storm,
        defaults={"min_error_ratio": 0.25, "min_reqs_per_min": 100.0},
    ),
    Rule(
        name="port_fanout",
        description="Unusual number of unique destination ports",
        fn=_rule_port_fanout,
        defaults={"min_unique_ports": 50.0},
    ),
    Rule(
        name="request_spike",
        description="Traffic spike (z-score on reqs_per_min)",
        fn=_rule_request_spike,
        defaults={"min_rpm_z": 3.0},
    ),
]

RULE_INDEX = {r.name: r for r in RULES}


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def apply_rule(df: pd.DataFrame, rule_name: str, overrides: Optional[Dict[str, float]] = None) -> pd.Series:
    """Apply a single rule by name and return a boolean mask."""
    if rule_name not in RULE_INDEX:
        raise KeyError(f"Unknown rule: {rule_name}")
    return RULE_INDEX[rule_name].run(df, overrides)


def apply_all(df: pd.DataFrame, *, cfg: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
    """Apply all rules and return an alerts DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    cfg : Optional mapping of rule_name -> overrides dict.

    Returns
    -------
    pd.DataFrame with columns:
        - rule: rule name
        - index: original row index
        - message: human-readable reason
    """
    rows: List[Dict[str, object]] = []
    for rule in RULES:
        overrides = (cfg or {}).get(rule.name)
        mask = rule.run(df, overrides)
        if not mask.any():
            continue
        idx = np.where(mask.values)[0]
        for i in idx:
            rows.append({
                "rule": rule.name,
                "index": df.index[i],
                "message": rule.description,
            })

    return pd.DataFrame(rows, columns=["rule", "index", "message"])
def detect_suspicious_activity(
    df: pd.DataFrame, *, cfg: Optional[Dict[str, Dict[str, float]]] = None
) -> pd.DataFrame:
    """Detect suspicious activity using all configured rules."""
    return apply_all(df, cfg=cfg)
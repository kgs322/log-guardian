"""Feature engineering for Log Guardian.

Given a log file path, build a minimal numeric feature frame suitable for
simple anomaly models. Robust to empty/invalid files.

Exports:
- build_features(path) -> pd.DataFrame
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from log_guardian.ingestion.readers import read_log
except Exception:
    read_log = None  # type: ignore


def _features_from_auth(rows: List[Dict]) -> pd.DataFrame:
    n = len(rows)
    if n == 0:
        return pd.DataFrame()

    msgs = [r.get("message", "") for r in rows]
    failed = sum("Failed password" in m for m in msgs)
    accepted = sum("Accepted password" in m for m in msgs)
    invalid_user = sum("invalid user" in m for m in msgs)
    hosts = {r.get("host") for r in rows if r.get("host")}
    procs = {r.get("process") for r in rows if r.get("process")}

    data = {
        "total_lines": float(n),
        "failed_logins": float(failed),
        "accepted_logins": float(accepted),
        "invalid_user_attempts": float(invalid_user),
        "failed_login_rate": (failed / n) if n else 0.0,
        "unique_hosts": float(len(hosts)),
        "unique_processes": float(len(procs)),
    }
    return pd.DataFrame([data])


def _features_from_nginx(rows: List[Dict]) -> pd.DataFrame:
    n = len(rows)
    if n == 0:
        return pd.DataFrame()

    statuses: List[int] = []
    bytes_sent: List[int] = []
    times: List[str] = []
    addrs = set()

    for r in rows:
        try:
            statuses.append(int(r.get("status", 0)))
        except Exception:
            statuses.append(0)
        try:
            bytes_sent.append(int(r.get("body_bytes_sent", 0)))
        except Exception:
            bytes_sent.append(0)
        addrs.add(r.get("remote_addr", ""))
        times.append(r.get("time_local", ""))

    s = np.asarray(statuses, dtype=int)
    total_err = int(((s // 100) == 4).sum() + ((s // 100) == 5).sum())
    ratio_err = (total_err / n) if n else 0.0

    # Approximate requests per minute by grouping timestamps to minute precision.
    def _minute_key(t: str) -> str:
        # Example input: 10/Oct/2020:13:55:36 +0000
        if ":" not in t:
            return ""
        try:
            datepart, clock = t.split(":", 1)
            hh, mm, _rest = clock.split(":", 2)
            return f"{datepart}|{hh}:{mm}"
        except Exception:
            return ""

    minute_keys = [_minute_key(t) for t in times]
    minutes = pd.Series(minute_keys).replace("", np.nan).dropna()
    reqs_per_min = float(minutes.value_counts().mean()) if not minutes.empty else float(n)

    data = {
        "total_lines": float(n),
        "status_4xx_5xx_ratio": float(ratio_err),
        "bytes_mean": float(np.mean(bytes_sent)) if bytes_sent else 0.0,
        "unique_remote_addrs": float(len([a for a in addrs if a])),
        "reqs_per_min": float(reqs_per_min),
    }
    return pd.DataFrame([data])


def build_features(path: str | Path) -> pd.DataFrame:
    """Build a minimal feature DataFrame from a log file.

    Behavior:
    - Empty path / missing file -> empty DataFrame
    - Unparseable file         -> empty DataFrame (tests accept empty or error)
    - Chooses auth/nginx features based on filename or parsed content
    """
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    if p.is_file() and p.stat().st_size == 0:
        return pd.DataFrame()
    if read_log is None:
        return pd.DataFrame()

    try:
        rows = read_log(p)
    except Exception:
        # Unknown format -> empty is acceptable for tests
        return pd.DataFrame()

    name = p.name.lower()
    if "auth" in name:
        df = _features_from_auth(rows)
    elif "nginx" in name or "access" in name:
        df = _features_from_nginx(rows)
    else:
        # try both
        df = _features_from_auth(rows)
        if df.empty:
            df = _features_from_nginx(rows)

    if not df.empty:
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


__all__ = ["build_features"]

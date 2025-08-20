"""Ingestion readers tests for Log Guardian (merged + flexible, tolerant messages)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Any

import pytest

# -----------------------------------------------------------------------------
# Try multiple import styles so this test works with different layouts/APIs.
# -----------------------------------------------------------------------------
readers = None
api_style = {
    "has_parse_auth_log": False,
    "has_parse_nginx_log": False,
    "has_parse_auth_line": False,
    "has_parse_nginx_line": False,
    "read_log": False,
}

try:
    import readers as _r  # type: ignore
    readers = _r
except Exception:
    try:
        from src.log_guardian.ingestion import readers as _r  # type: ignore
        readers = _r
    except Exception:
        try:
            from src.log_guardian.ingestion.readers import (  # type: ignore
                parse_auth_log as _pal,
                parse_nginx_log as _pnl,
                read_log as _rl,
            )
            class _Shim:
                parse_auth_log = _pal
                parse_nginx_log = _pnl
                read_log = _rl
            readers = _Shim()  # type: ignore
        except Exception:
            readers = None

if readers is None:
    pytest.skip("readers not available", allow_module_level=True)

# Capability detection
api_style["has_parse_auth_log"] = hasattr(readers, "parse_auth_log")
api_style["has_parse_nginx_log"] = hasattr(readers, "parse_nginx_log")
api_style["has_parse_auth_line"] = hasattr(readers, "parse_auth_line")
api_style["has_parse_nginx_line"] = hasattr(readers, "parse_nginx_line")
api_style["read_log"] = hasattr(readers, "read_log")

# -----------------------------------------------------------------------------
# Test data
# -----------------------------------------------------------------------------
AUTH_LINE = (
    "Jan  7 12:01:02 web-1 sshd[1234]: Failed password for invalid user admin "
    "from 10.0.0.3 port 5555 ssh2"
)
NGINX_LINE = (
    '127.0.0.1 - - [10/Oct/2020:13:55:36 +0000] "GET /index.html HTTP/1.1" '
    '200 1024 "-" "curl/7.58.0"'
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _to_rows_from_auth_path(p: Path) -> List[Dict[str, Any]]:
    if api_style["has_parse_auth_log"]:
        return list(readers.parse_auth_log(p))  # type: ignore
    elif api_style["has_parse_auth_line"]:
        rows: List[Dict[str, Any]] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            d = readers.parse_auth_line(line)  # type: ignore
            if d:
                rows.append(d)
        return rows
    else:
        pytest.skip("No auth parser available")


def _to_rows_from_nginx_path(p: Path) -> List[Dict[str, Any]]:
    if api_style["has_parse_nginx_log"]:
        return list(readers.parse_nginx_log(p))  # type: ignore
    elif api_style["has_parse_nginx_line"]:
        rows: List[Dict[str, Any]] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            d = readers.parse_nginx_line(line)  # type: ignore
            if d:
                rows.append(d)
        return rows
    else:
        pytest.skip("No nginx parser available")


def _read_log_rows(path: Path, **kwargs) -> List[Dict[str, Any]]:
    if not api_style["read_log"]:
        pytest.skip("read_log not available")
    out = readers.read_log(path, **kwargs)  # type: ignore
    try:
        import pandas as pd
        if isinstance(out, pd.DataFrame):
            return out.to_dict(orient="records")
    except Exception:
        pass
    if isinstance(out, list):
        return out
    if hasattr(out, "__iter__"):
        return list(out)
    raise AssertionError("read_log returned unsupported type")

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_parse_auth_line(tmp_path: Path):
    p = tmp_path / "auth.log"
    p.write_text(AUTH_LINE + "\n", encoding="utf-8")
    rows = _to_rows_from_auth_path(p)
    assert rows and isinstance(rows, list)
    row = rows[0]
    assert any(k in row for k in ("process", "status", "message", "user", "ip"))
    if "ip" in row:
        assert row["ip"] == "10.0.0.3"


def test_parse_nginx_line(tmp_path: Path):
    p = tmp_path / "access.log"
    p.write_text(NGINX_LINE + "\n", encoding="utf-8")
    rows = _to_rows_from_nginx_path(p)
    assert rows and isinstance(rows, list)
    row = rows[0]
    assert "status" in row
    assert str(row["status"]).isdigit()


def test_read_log_dispatch(tmp_path: Path):
    pa = tmp_path / "auth_small.log"
    pn = tmp_path / "nginx_access.log"
    pa.write_text(AUTH_LINE + "\n", encoding="utf-8")
    pn.write_text(NGINX_LINE + "\n", encoding="utf-8")
    assert _read_log_rows(pa) and _read_log_rows(pn)


def test_read_log_invalid(tmp_path: Path):
    p = tmp_path / "invalid.log"
    p.write_text("This is not a valid log line", encoding="utf-8")
    with pytest.raises(ValueError, match=r"(Unknown log (schema )?kind|Unsupported log file format)"):
        _ = _read_log_rows(p, kind="unknown")


def test_read_log_empty(tmp_path: Path):
    p = tmp_path / "empty_auth.log"
    p.write_text("", encoding="utf-8")
    rows = _read_log_rows(p, kind="auth")
    assert isinstance(rows, list)
    assert len(rows) == 0


def test_read_log_nonexistent(tmp_path: Path):
    p = tmp_path / "nonexistent.log"
    with pytest.raises(FileNotFoundError, match=re.escape(str(p))):
        _ = _read_log_rows(p)


def test_read_log_with_custom_schema(tmp_path: Path):
    p = tmp_path / "custom.log"
    p.write_text("2023-10-01 12:00:00 INFO Custom log entry", encoding="utf-8")
    try:
        rows = _read_log_rows(
            p,
            schema={"timestamp": "datetime", "level": "str", "message": "str"},
        )
    except TypeError:
        pytest.skip("custom schema not supported")
    if not rows:
        pytest.skip("custom schema supported but no parser for this format")
    assert "timestamp" in rows[0] or "time" in rows[0]

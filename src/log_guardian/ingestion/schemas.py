"""Row schemas for parsed logs in Log Guardian.

Defines **Pydantic** models used to validate structured rows parsed from
raw log lines. Keep this module focused on *schemas only* — do not mix in
parsing/reader code here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

__all__ = [
    "AuthLogRow",
    "NginxLogRow",
    "LOG_SCHEMAS",
    "validate_log_row",
]

# ------------------------------------------------------------------
# Auth log schema
# ------------------------------------------------------------------
class AuthLogRow(BaseModel):
    month: str = Field(..., description="Month abbreviation, e.g., 'Jan'")
    day: str = Field(..., description="Day of month as string (e.g., '7', '27')")
    time: str = Field(..., description="Time of log entry, HH:MM:SS")
    host: str = Field(..., description="Hostname")
    process: str = Field(..., description="Process name (e.g., 'sshd')")
    pid: Optional[str] = Field(None, description="Process ID if present")
    message: str = Field(..., description="Log message body")

    class Config:
        extra = "ignore"


# ------------------------------------------------------------------
# Nginx access log schema
# ------------------------------------------------------------------
class NginxLogRow(BaseModel):
    remote_addr: str
    remote_user: str
    time_local: str
    request: str
    status: str  # keep as str; convert later in features
    body_bytes_sent: str  # keep as str; convert later in features
    http_referer: str
    http_user_agent: str

    class Config:
        extra = "ignore"


# ------------------------------------------------------------------
# Registry & helpers
# ------------------------------------------------------------------
LOG_SCHEMAS: Dict[str, Any] = {
    "auth": AuthLogRow,
    "nginx": NginxLogRow,
}


def validate_log_row(kind: str, row: Dict[str, Any]) -> BaseModel:
    """Validate a parsed row according to schema kind.

    Parameters
    ----------
    kind: One of 'auth' or 'nginx'.
    row: Dict of parsed fields from a reader.

    Returns
    -------
    pydantic.BaseModel
        The validated model instance (AuthLogRow or NginxLogRow).
    """
    try:
        schema = LOG_SCHEMAS[kind]
    except KeyError as e:
        raise ValueError(f"Unknown log schema kind: {kind}") from e
    return schema(**row)
    """Read a log file and return structured records.
    Parameters
    ----------"""
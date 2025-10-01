#!/usr/bin/env python3
from __future__ import annotations

from src.utils.data.pids import normalize_pid


def pid_display_label(pid: str) -> str:
    """Return PID label substring after canonical AMLEC prefix, keeping any suffix."""
    normalized = normalize_pid(pid)
    if normalized:
        return normalized[len("AMLEC_") :]
    return str(pid)

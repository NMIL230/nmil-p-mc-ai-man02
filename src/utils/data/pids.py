#!/usr/bin/env python3
"""Helpers for parsing and normalizing AMLEC participant identifiers."""

from __future__ import annotations

import re
from typing import Optional, Sequence, List

PID_PATTERN = re.compile(r"^AMLEC[-_](\d+)((?:[-_][A-Za-z0-9]+)*)$")


def normalize_pid(raw: Optional[str]) -> Optional[str]:
    """Return canonical AMLEC PID using underscores for separators and upper-case suffix."""
    if raw is None:
        return None
    candidate = str(raw).strip()
    if not candidate:
        return None
    match = PID_PATTERN.match(candidate)
    if not match:
        return None
    digits = match.group(1)
    suffix = match.group(2) or ""
    normalized_suffix = re.sub(r"[-_]+", "_", suffix).upper()
    return f"AMLEC_{digits}{normalized_suffix}"


def is_canonical_pid(name: str) -> bool:
    """True when the provided name already matches the canonical AMLEC format."""
    return normalize_pid(name) == name


def base_pid(raw: Optional[str]) -> Optional[str]:
    """Return canonical participant ID without repeat suffix (AMLEC_xxx)."""
    normalized = normalize_pid(raw)
    if not normalized:
        return None
    parts = normalized.split("_")
    if len(parts) < 2:
        return normalized
    return f"{parts[0]}_{parts[1]}"


def repeat_index(raw: Optional[str]) -> Optional[int]:
    """Return repeat index (R#) when present, otherwise None."""
    normalized = normalize_pid(raw)
    if not normalized or "_R" not in normalized:
        return None
    suffix = normalized.rsplit("_R", 1)[-1]
    if suffix.isdigit():
        return int(suffix)
    return None


def is_repeat_pid(raw: Optional[str]) -> bool:
    """True when the PID corresponds to a repeat session (has _R# suffix)."""
    return repeat_index(raw) is not None


def collapse_repeat_sessions(pids: Sequence[str]) -> List[str]:
    """Return one PID per participant, preferring non-repeat sessions when available.

    The original ordering of participants is preserved based on their first
    appearance. If only repeat sessions are available for a participant, the
    lowest-numbered repeat is retained.
    """

    selected: dict[str, str] = {}
    base_order: List[str] = []

    for pid in pids:
        base = base_pid(pid) or normalize_pid(pid) or pid
        if base not in selected:
            selected[base] = pid
            base_order.append(base)
            continue

        existing = selected[base]
        existing_repeat = is_repeat_pid(existing)
        candidate_repeat = is_repeat_pid(pid)

        if existing_repeat and not candidate_repeat:
            selected[base] = pid
            continue

        if existing_repeat and candidate_repeat:
            existing_idx = repeat_index(existing)
            candidate_idx = repeat_index(pid)
            if candidate_idx is not None and (
                existing_idx is None or candidate_idx < existing_idx
            ):
                selected[base] = pid

    return [selected[base] for base in base_order]

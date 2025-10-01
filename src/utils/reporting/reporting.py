#!/usr/bin/env python3
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Optional
import numpy as np
from src.utils.data.pairing import filter_valid_psi_pairs
from src.utils.reporting.formatting import format_stat
from src.utils.plotting.constants import PSI_THETA_TEX


__all__ = [
    "validity_removed_pids",
    "format_validity_removed_line",
    "format_validity_details",
    "iqr_bounds",
    "filter_by_iqr_diff",
    "format_global_iqr_line",
    "format_iqr_removed_pids_line",
    "format_df_value",
    "format_df_pair",
    "format_ci_interval",
]


def validity_removed_pids(
    ent_pairs: Sequence[Tuple[str, float, float]],
    valid_ent_pairs: Sequence[Tuple[str, float, float]],
) -> List[str]:
    all_ids = [pid for (pid, _x, _y) in ent_pairs]
    kept_ids = {pid for (pid, _x, _y) in valid_ent_pairs}
    return [pid for pid in all_ids if pid not in kept_ids]


def format_validity_removed_line(removed_pids: Sequence[str]) -> Optional[str]:
    if not removed_pids:
        return None
    ordered = sorted(map(str, removed_pids))
    return f"Validity-removed PIDs ({len(ordered)}): " + ", ".join(ordered)


def format_validity_details(
    ent_pairs: Sequence[Tuple[str, float, float]],
    *,
    min_allowed: float = 0.0,
    max_allowed: float = 18.0,
    max_detail: int = 20,
) -> Optional[str]:
    """Return detailed lines for invalid pairs: PID, mode, psi, and reason.

    Uses `filter_valid_psi_pairs` to recompute validity and collect example reasons.
    """
    try:
        _pairs, _labels, _valid, report = filter_valid_psi_pairs(
            ent_pairs,
            min_allowed=min_allowed,
            max_allowed=max_allowed,
            max_detail=max_detail,
        )
    except Exception:
        return None
    skipped = report.get("skipped_examples", []) if isinstance(report, dict) else []
    if not skipped:
        return None
    lines: List[str] = []
    min_fmt = format_stat(min_allowed, mode="decimal")
    max_fmt = format_stat(max_allowed, mode="decimal")
    lines.append(f"Validity details (bounds [{min_fmt}, {max_fmt}]):")
    for ex in skipped:
        pid = str(ex.get("participant", ""))
        label = str(ex.get("label", pid))
        classic = ex.get("classic")
        adaptive = ex.get("adaptive")
        reasons = [str(r) for r in ex.get("reasons", [])]
        # Build a compact per-PID line
        vals = []
        if classic is not None and np.isfinite(classic):
            vals.append(
                f"Classic {PSI_THETA_TEX}={format_stat(float(classic), mode='decimal')}"
            )
        else:
            vals.append(f"Classic {PSI_THETA_TEX}=NA")
        if adaptive is not None and np.isfinite(adaptive):
            vals.append(
                f"Adaptive {PSI_THETA_TEX}={format_stat(float(adaptive), mode='decimal')}"
            )
        else:
            vals.append(f"Adaptive {PSI_THETA_TEX}=NA")
        reasons_text = "; ".join(reasons) if reasons else ""
        lines.append(f"- {label} [{pid}]: {', '.join(vals)} | {reasons_text}")
    return "\n".join(lines)


def format_df_value(value: float | int | None) -> str:
    """Return a consistent string for degrees of freedom counts."""
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "NA"
    if abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))
    return format_stat(numeric, mode="decimal")


def format_df_pair(df1: float | int | None, df2: float | int | None) -> str:
    """Format a pair of degrees-of-freedom values."""
    return f"({format_df_value(df1)}, {format_df_value(df2)})"


def format_ci_interval(
    lo: float | None,
    hi: float | None,
    *,
    confidence: int = 95,
) -> str:
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{confidence}% CI [NA, NA]"
    lo_txt = format_stat(float(lo), mode="decimal")
    hi_txt = format_stat(float(hi), mode="decimal")
    return f"{confidence}% CI [{lo_txt}, {hi_txt}]"


def iqr_bounds(values: Iterable[float], multiplier: float = 1.5) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    return float(q1 - multiplier * iqr), float(q3 + multiplier * iqr)


def filter_by_iqr_diff(
    valid_ent_pairs: Sequence[Tuple[str, float, float]],
    lo: float,
    hi: float,
) -> Tuple[List[Tuple[str, float, float]], List[str]]:
    filtered: List[Tuple[str, float, float]] = []
    removed_pids: List[str] = []
    for pid, x, y in valid_ent_pairs:
        d = float(y) - float(x)
        if np.isfinite(d) and (d >= lo) and (d <= hi):
            filtered.append((pid, float(x), float(y)))
        else:
            removed_pids.append(str(pid))
    return filtered, removed_pids


def format_global_iqr_line(
    removed_count: int,
    total_count: int,
    lo: float,
    hi: float,
) -> str:
    lo_fmt = format_stat(lo, mode="decimal")
    hi_fmt = format_stat(hi, mode="decimal")
    return f"Global IQR filter: removed {removed_count}/{total_count} diffs (bounds {lo_fmt} to {hi_fmt})"


def format_iqr_removed_pids_line(removed_pids: Sequence[str]) -> Optional[str]:
    if not removed_pids:
        return None
    ordered = sorted(map(str, removed_pids))
    return "IQR-removed PIDs: " + ", ".join(ordered)

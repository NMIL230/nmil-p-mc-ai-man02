#!/usr/bin/env python3
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any

import numpy as np

from src.utils.plotting.plotting import (
    CORR_OUTLIER_METHOD_DEFAULT,
    CORR_IQR_MULTIPLIER_DEFAULT,
    CORR_ZSCORE_THRESHOLD_DEFAULT,
)


def compute_outlier_mask_on_diffs(
    diffs: np.ndarray,
    *,
    method: str = CORR_OUTLIER_METHOD_DEFAULT,
    iqr_multiplier: float = CORR_IQR_MULTIPLIER_DEFAULT,
    z_thresh: float = CORR_ZSCORE_THRESHOLD_DEFAULT,
) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    """Return boolean mask of kept items and optional bounds for reporting.

    - method "iqr_diff": keep within IQR*mult bounds of diffs
    - method "zscore_diff": keep within ±z_thresh SD around mean diff
    - method "none" or unknown: keep all
    """
    diffs = np.asarray(diffs, dtype=float)
    bounds: Optional[Tuple[float, float]] = None
    m = str(method).lower().strip()
    if diffs.size == 0 or not np.any(np.isfinite(diffs)):
        return np.ones_like(diffs, dtype=bool), None
    diffs = diffs[np.isfinite(diffs)]
    if m == "iqr_diff":
        q1, q3 = np.percentile(diffs, [25, 75])
        iqr = q3 - q1
        lo, hi = (q1 - iqr_multiplier * iqr), (q3 + iqr_multiplier * iqr)
        bounds = (float(lo), float(hi))
        mask = (diffs >= lo) & (diffs <= hi)
    elif m == "zscore_diff":
        mu = float(np.mean(diffs))
        sd = float(np.std(diffs, ddof=1) if diffs.size > 1 else np.std(diffs))
        if np.isfinite(sd) and sd > 0:
            bounds = (mu - z_thresh * sd, mu + z_thresh * sd)
        z = (diffs - mu) / (sd if sd > 0 else 1.0)
        mask = np.abs(z) <= z_thresh
    else:
        mask = np.ones_like(diffs, dtype=bool)
        bounds = None
    return mask, bounds


def filter_df_by_diff_column(
    df, diff_column: str,
    *,
    method: str = CORR_OUTLIER_METHOD_DEFAULT,
    iqr_multiplier: float = CORR_IQR_MULTIPLIER_DEFAULT,
    z_thresh: float = CORR_ZSCORE_THRESHOLD_DEFAULT,
) -> Tuple[Any, Dict[str, Any]]:
    """Filter a DataFrame by keeping rows whose `diff_column` is within bounds.

    Returns (filtered_df, info) where info includes counts and bounds.
    """
    vals = np.asarray(df[diff_column], dtype=float)
    finite_mask = np.isfinite(vals)
    diffs = vals[finite_mask]
    keep_mask_local, bounds = compute_outlier_mask_on_diffs(diffs, method=method, iqr_multiplier=iqr_multiplier, z_thresh=z_thresh)
    # Map local mask back to full-length mask
    keep_full = np.zeros_like(finite_mask, dtype=bool)
    idxs = np.where(finite_mask)[0]
    keep_full[idxs] = keep_mask_local
    filtered_df = df[keep_full].copy()
    info = {
        "before": int(len(df)),
        "after": int(len(filtered_df)),
        "removed": int(len(df) - len(filtered_df)),
        "bounds": bounds,
        "method": method,
        "column": diff_column,
    }
    return filtered_df, info


def filter_pids_by_pair_diffs(
    valid_ent_pairs: Sequence[Tuple[str, float, float]],
    *,
    method: str = CORR_OUTLIER_METHOD_DEFAULT,
    iqr_multiplier: float = CORR_IQR_MULTIPLIER_DEFAULT,
    z_thresh: float = CORR_ZSCORE_THRESHOLD_DEFAULT,
) -> Tuple[set, Dict[str, Any]]:
    """Given valid (pid, x, y) pairs, compute y-x diffs and return kept PID set.

    Returns (kept_pids, info) where info includes bounds and counts.
    """
    if not valid_ent_pairs:
        return set(), {"before": 0, "after": 0, "removed": 0, "bounds": None, "method": method}
    pids: List[str] = []
    diffs: List[float] = []
    for pid, x_val, y_val in valid_ent_pairs:
        try:
            dx = float(x_val)
            dy = float(y_val)
            if np.isfinite(dx) and np.isfinite(dy):
                pids.append(str(pid))
                diffs.append(dy - dx)
        except Exception:
            continue
    diffs_arr = np.asarray(diffs, dtype=float)
    keep_mask, bounds = compute_outlier_mask_on_diffs(diffs_arr, method=method, iqr_multiplier=iqr_multiplier, z_thresh=z_thresh)
    kept_pids = {pids[i] for i in range(len(pids)) if keep_mask[i]}
    info = {
        "before": len(pids),
        "after": len(kept_pids),
        "removed": len(pids) - len(kept_pids),
        "bounds": bounds,
        "method": method,
    }
    return kept_pids, info


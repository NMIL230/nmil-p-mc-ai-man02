#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data.common_inputs import build_fits_and_pairs
from src.utils.statistics.outliers import filter_pids_by_pair_diffs


@dataclass
class ModeCounts:
    label: str
    per_pid: Dict[str, float]

    def to_array(self) -> np.ndarray:
        if not self.per_pid:
            return np.array([], dtype=float)
        return np.array(list(self.per_pid.values()), dtype=float)


def collect_classic_counts(
    fits_by_pid: Mapping[str, List[Dict[str, float]]],
    pids: Iterable[str],
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for pid in pids:
        fit_list = fits_by_pid.get(pid, [])
        total = 0.0
        for fit in fit_list:
            dm = fit.get("data")
            if not isinstance(dm, dict):
                continue
            for stats in dm.values():
                if not isinstance(stats, dict):
                    continue
                trials = float(stats.get("trials", 0.0))
                if math.isfinite(trials) and trials > 0.0:
                    total += trials
        if total > 0.0:
            totals[str(pid)] = total
    return totals


def collect_adaptive_counts(
    raw_points_by_pid: Mapping[str, List[Tuple[float, float, int]]],
    pids: Iterable[str],
    *,
    k_value: Optional[float] = None,
    tol: float = 1e-9,
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for pid in pids:
        pts = raw_points_by_pid.get(pid, []) or []
        if k_value is None:
            count = len(pts)
        else:
            count = sum(1 for _diff, color_count, _outcome in pts if abs(float(color_count) - k_value) <= tol)
        if count > 0:
            totals[str(pid)] = float(count)
    return totals


def summarize_counts(mode: ModeCounts) -> Dict[str, float]:
    arr = mode.to_array()
    if arr.size == 0:
        return {}
    return {
        "pids": int(arr.size),
        "total": float(arr.sum()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def format_summary(summary: Dict[str, float], label: str) -> str:
    if not summary:
        return f"{label}: no data"
    return (
        f"{label}: pids={int(summary['pids'])}, total={summary['total']:.0f}, "
        f"mean={summary['mean']:.2f}, std={summary['std']:.2f}, "
        f"range=({summary['min']:.0f}, {summary['max']:.0f})"
    )


def determine_pid_set(
    ent_pairs: Sequence[Tuple[str, float, float]],
    valid_ent_pairs: Sequence[Tuple[str, float, float]],
    *,
    apply_validity_mask: bool,
    apply_outlier_filter: bool,
    outlier_method: str,
    iqr_multiplier: float,
    z_thresh: float,
) -> Tuple[set[str], Dict[str, float]]:
    base_pairs = valid_ent_pairs if apply_validity_mask else ent_pairs
    pid_set = {pid for (pid, _x, _y) in base_pairs}
    info: Dict[str, float] = {}
    if apply_outlier_filter and base_pairs:
        kept_pids, filt_info = filter_pids_by_pair_diffs(
            base_pairs,
            method=outlier_method,
            iqr_multiplier=iqr_multiplier,
            z_thresh=z_thresh,
        )
        info = {k: v for k, v in filt_info.items() if isinstance(v, (int, float, str))}
        if kept_pids:
            pid_set &= kept_pids
    return pid_set, info


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-participant trial counts for Classic and Adaptive modes",
    )
    parser.add_argument("--pickle", type=Path, default=None, help="Optional path to AMLEC dataset pickle")
    parser.add_argument("--no-validity-mask", dest="apply_validity_mask", action="store_false", help="Disable validity mask (default ON)")
    parser.add_argument("--validity-mask", dest="apply_validity_mask", action="store_true", help="Enable validity mask")
    parser.set_defaults(apply_validity_mask=True)
    parser.add_argument("--no-outlier-filter", dest="apply_outlier_filter", action="store_false", help="Skip modality diff outlier filtering (default ON)")
    parser.add_argument("--outlier-filter", dest="apply_outlier_filter", action="store_true", help="Enable modality diff outlier filtering")
    parser.set_defaults(apply_outlier_filter=True)
    parser.add_argument("--outlier-method", type=str, default="iqr_diff", choices=["iqr_diff", "zscore_diff", "none"], help="Outlier method for modality diffs")
    parser.add_argument("--iqr-multiplier", type=float, default=1.5, help="Multiplier for IQR fence (iqr_diff)")
    parser.add_argument("--z-thresh", type=float, default=3.0, help="Z-score threshold (zscore_diff)")
    parser.add_argument("--k", type=float, default=None, help="If provided, filter adaptive counts to this color-count K value")
    parser.add_argument("--detail", action="store_true", help="Print per-PID counts as well as summary stats")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    pickle_path = Path(args.pickle) if args.pickle is not None else None

    fx, fy, ent_pairs, _pairs, valid_ent_pairs, _labels, gp_by_pid, adaptive_raw_points_all = build_fits_and_pairs(pickle_path)

    pid_set, filt_info = determine_pid_set(
        ent_pairs,
        valid_ent_pairs,
        apply_validity_mask=args.apply_validity_mask,
        apply_outlier_filter=args.apply_outlier_filter and args.outlier_method != "none",
        outlier_method=args.outlier_method,
        iqr_multiplier=args.iqr_multiplier,
        z_thresh=args.z_thresh,
    )

    if not pid_set:
        print("No participants remain after filtering; nothing to summarize.")
        return

    classic_counts = collect_classic_counts(fx, pid_set)
    adaptive_counts = collect_adaptive_counts(
        adaptive_raw_points_all,
        pid_set,
        k_value=args.k,
    )

    classic_summary = summarize_counts(ModeCounts("Classic", classic_counts))
    adaptive_summary = summarize_counts(ModeCounts("Adaptive", adaptive_counts))

    print(format_summary(classic_summary, "Classic"))
    if args.k is None:
        print(format_summary(adaptive_summary, "Adaptive"))
    else:
        print(format_summary(adaptive_summary, f"Adaptive (K={args.k:g})"))

    if args.detail:
        print("\nPer-PID counts (Classic):")
        for pid, total in sorted(classic_counts.items()):
            print(f"  {pid}: {total:.0f}")
        print("\nPer-PID counts (Adaptive):")
        for pid, total in sorted(adaptive_counts.items()):
            print(f"  {pid}: {total:.0f}")

    if filt_info:
        removed = int(filt_info.get("removed", 0))
        before = int(filt_info.get("before", 0))
        after = int(filt_info.get("after", len(pid_set)))
        method = filt_info.get("method")
        bounds = filt_info.get("bounds")
        msg = f"Outlier filter ({method}) removed {removed}/{before}; kept {after}"
        if isinstance(bounds, tuple) and len(bounds) == 2:
            msg += f"; bounds={bounds[0]:.2f}..{bounds[1]:.2f}"
        print(msg)


if __name__ == "__main__":
    main()

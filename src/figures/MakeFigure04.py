#!/usr/bin/env python3
r"""Correlation figure for paired Classic vs Adaptive $\psi_{\theta}$ with ICC and CI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence
import numpy as np

# Ensure repo root on sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plotting.constants import PSI_THETA_TEX
from src.utils.data.common_inputs import build_fits_and_pairs
from src.utils.statistics.corr import correlate_psis
from src.utils.reporting.reporting import (
    validity_removed_pids,
    format_validity_removed_line,
    format_global_iqr_line,
    format_iqr_removed_pids_line,
    format_validity_details,
    format_ci_interval,
    format_df_value,
    format_df_pair,
)
from src.utils.data.parity import (
    build_parity_records,
    parity_dataframe,
    estimate_global_session_effect,
)
from src.data.pid_order import determine_mode_order_for_pid
from src.utils.reporting.formatting import format_stat
from src.utils.data.pids import normalize_pid, collapse_repeat_sessions
from src.utils.plotting.figure_naming import figure_output_dir, figure_output_stem

FIGURE_KEY = "Fig04"
FIGURE_STEM = figure_output_stem(FIGURE_KEY)
DEFAULT_FIGURE_OUTPUT = figure_output_dir(FIGURE_KEY)
FIGURE_NAME = DEFAULT_FIGURE_OUTPUT.name


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate correlation ICC figure")
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_FIGURE_OUTPUT,
        help="Output directory",
    )
    p.add_argument("--pickle", type=Path, default=None, help="AMLEC pickle path (optional)")
    # Plot + stats controls
    p.add_argument("--filename-stem", type=str, default=FIGURE_STEM, help="Output filename stem")
    p.add_argument("--equal-axes", dest="equal_axes", action="store_true", help="Force equal axes limits (default)")
    p.add_argument("--unequal-axes", dest="equal_axes", action="store_false", help="Allow independent axis limits")
    p.set_defaults(equal_axes=True)
    # Outlier handling (defaults to exclusion on)
    p.add_argument("--no-exclude-outliers", dest="exclude_outliers", action="store_false", help="Keep all points; disable outlier exclusion")
    p.add_argument("--exclude-outliers", dest="exclude_outliers", action="store_true", help="Enable outlier exclusion (default)")
    p.set_defaults(exclude_outliers=True)
    p.add_argument(
        "--outlier-method",
        choices=["iqr_diff", "iqr", "zscore_diff", "zscore"],
        default="iqr_diff",
        help="Outlier detection method",
    )
    p.add_argument("--iqr-multiplier", type=float, default=1.5, help="IQR fence multiplier (for IQR methods)")
    p.add_argument("--z-thresh", type=float, default=3.0, help="Z-score threshold (for zscore methods)")
    # Optional global session order adjustment
    p.add_argument("--adjust-session-effect", dest="adjust_session_effect", action="store_true", help="Regress out global second-vs-first session effect (default)")
    p.add_argument("--no-adjust-session-effect", dest="adjust_session_effect", action="store_false", help="Disable session effect adjustment")
    p.set_defaults(adjust_session_effect=True)
    # Fixed axis limits (two floats) or disable
    p.add_argument("--fixed-limits", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="Set fixed axis limits (min max)")
    p.add_argument("--no-fixed-limits", dest="fixed_limits", action="store_const", const=None, help="Disable fixed axis limits")
    # Labels
    p.add_argument("--show-labels", dest="show_labels", action="store_true", help="Show point labels (default)")
    p.add_argument("--hide-labels", dest="show_labels", action="store_false", help="Hide point labels")
    p.set_defaults(show_labels=False)
    p.add_argument("--exclude-repeats", action="store_true", help="Exclude participants with _R# repeat suffix (e.g., AMLEC_018_R1)")
    p.add_argument(
        "--stats-config",
        type=str,
        default=None,
        help="JSON array (or path) of format strings for correlation statistics.",
    )
    return p.parse_args(argv)


def _load_stats_config(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.exists():
        text = candidate.read_text(encoding="utf-8")
    else:
        text = raw
    config = json.loads(text)
    if isinstance(config, dict) and "lines" in config:
        config = config["lines"]
    if isinstance(config, str):
        return [config]
    if not isinstance(config, list):
        raise ValueError("stats-config must be a JSON array or object with 'lines'.")
    return [str(item) for item in config]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    _fx, _fy, ent_pairs, pairs, valid_ent_pairs, point_labels, _gp, _pts = build_fits_and_pairs(args.pickle)
    # Report validity filtering consistently
    removed_valid = validity_removed_pids(ent_pairs, valid_ent_pairs)
    line = format_validity_removed_line(removed_valid)
    if line:
        print(line)
        details = format_validity_details(ent_pairs, min_allowed=0.0, max_allowed=18.0)
        if details:
            print(details)
    # Mirror mask summary for identical printouts
    before = len(ent_pairs)
    after = len(valid_ent_pairs)
    if before != after:
        print(f"Applied validity mask: removed {before - after}/{before} subjects; kept {after}")
    ent_source = list(valid_ent_pairs)
    labels_list = list(point_labels) if point_labels is not None else None

    pairs_array = np.array([(float(x), float(y)) for (_pid, x, y) in ent_source], dtype=float)
    kept_indices = np.arange(len(ent_source))
    outlier_bounds = None
    did_outlier_filter = False
    if args.exclude_outliers and len(ent_source) >= 3:
        did_outlier_filter = True
        x_vals = pairs_array[:, 0]
        y_vals = pairs_array[:, 1]
        method = args.outlier_method
        mask = np.ones(len(ent_source), dtype=bool)
        if method == "iqr_diff":
            diffs = y_vals - x_vals
            q1, q3 = np.percentile(diffs, [25, 75])
            iqr = q3 - q1
            lo, hi = (q1 - args.iqr_multiplier * iqr), (q3 + args.iqr_multiplier * iqr)
            outlier_bounds = (float(lo), float(hi))
            mask = (diffs >= lo) & (diffs <= hi)
        elif method == "iqr":
            def _bounds(arr: np.ndarray) -> tuple[float, float]:
                q1, q3 = np.percentile(arr, [25, 75])
                iqr = q3 - q1
                return (q1 - args.iqr_multiplier * iqr), (q3 + args.iqr_multiplier * iqr)
            x_lo, x_hi = _bounds(x_vals)
            y_lo, y_hi = _bounds(y_vals)
            mask = (x_vals >= x_lo) & (x_vals <= x_hi) & (y_vals >= y_lo) & (y_vals <= y_hi)
        elif method == "zscore_diff":
            diffs = y_vals - x_vals
            mean = diffs.mean()
            sd = diffs.std(ddof=1) if len(diffs) > 1 else diffs.std()
            if np.isfinite(sd) and sd > 0:
                outlier_bounds = (float(mean - args.z_thresh * sd), float(mean + args.z_thresh * sd))
                z = (diffs - mean) / sd
            else:
                z = np.zeros_like(diffs)
            mask = np.abs(z) <= args.z_thresh
        elif method == "zscore":
            def _z(arr: np.ndarray) -> np.ndarray:
                sd = arr.std(ddof=1) if len(arr) > 1 else arr.std()
                if np.isfinite(sd) and sd > 0:
                    return (arr - arr.mean()) / sd
                return np.zeros_like(arr)
            zx = _z(x_vals)
            zy = _z(y_vals)
            mask = (np.abs(zx) <= args.z_thresh) & (np.abs(zy) <= args.z_thresh)
        kept_indices = np.where(mask)[0]

    filtered_ent_pairs = [ent_source[i] for i in kept_indices]
    filtered_labels = [labels_list[i] for i in kept_indices] if labels_list is not None else None

    removed_count = len(ent_source) - len(filtered_ent_pairs)
    if did_outlier_filter and removed_count > 0 and outlier_bounds is not None:
        lo, hi = outlier_bounds
        print(format_global_iqr_line(removed_count, len(ent_source), float(lo), float(hi)))
        kept_set = set(map(int, np.array(kept_indices, dtype=int)))
        removed_pids = [ent_source[i][0] for i in range(len(ent_source)) if i not in kept_set]
        line2 = format_iqr_removed_pids_line(removed_pids)
        if line2:
            print(line2)

    if args.exclude_repeats and filtered_ent_pairs:
        filtered_pids = [pid for (pid, _x, _y) in filtered_ent_pairs]
        collapsed_pids = collapse_repeat_sessions(filtered_pids)
        selected_indices: list[int] = []
        used_indices: set[int] = set()
        for pid in collapsed_pids:
            for idx, candidate in enumerate(filtered_pids):
                if idx in used_indices:
                    continue
                if candidate == pid:
                    selected_indices.append(idx)
                    used_indices.add(idx)
                    break
        selected_indices.sort()
        if len(selected_indices) != len(filtered_ent_pairs):
            removed_repeats = len(filtered_ent_pairs) - len(selected_indices)
            if removed_repeats > 0:
                print(f"Collapsed repeats: removed {removed_repeats} session(s) in favour of non-repeat data.")
        filtered_ent_pairs = [filtered_ent_pairs[i] for i in selected_indices]
        if filtered_labels is not None:
            filtered_labels = [filtered_labels[i] for i in selected_indices]

    # Optional: regress out global session 2 effect before building pairs
    pairs_for_corr = [(x, y) for (_pid, x, y) in filtered_ent_pairs]
    did_session_adjust = False
    if args.adjust_session_effect and len(filtered_ent_pairs) >= 3:
        try:
            records = build_parity_records()
            df_parity = parity_dataframe(records)
            valid_pids = {pid for (pid, _x, _y) in filtered_ent_pairs}
            df_parity = df_parity[df_parity["pid"].isin(valid_pids)].copy()
            beta = estimate_global_session_effect(df_parity)
            if abs(beta) > 0:
                beta_str = format_stat(beta, mode="decimal")
                if beta >= 0:
                    beta_str = f"+{beta_str}"
                print(f"Estimated global session effect (second vs first): {beta_str}; removing from second-session {PSI_THETA_TEX}")
            adjusted_ent_pairs = []
            for pid, x_val, y_val in filtered_ent_pairs:
                order = determine_mode_order_for_pid(pid)
                if order == "adaptive_first":
                    adjusted_ent_pairs.append((pid, float(x_val) - float(beta), float(y_val)))
                else:
                    adjusted_ent_pairs.append((pid, float(x_val), float(y_val) - float(beta)))
            pairs_for_corr = [(x, y) for (_pid, x, y) in adjusted_ent_pairs]
            did_session_adjust = True
        except Exception:
            did_session_adjust = False

    if did_outlier_filter and did_session_adjust:
        print("Order: applied outlier exclusion BEFORE session-order adjustment and correlation.")
    elif did_outlier_filter:
        print("Order: applied outlier exclusion; no session-order adjustment requested.")
    elif did_session_adjust:
        print("Order: applied session-order adjustment without additional outlier exclusion.")
    else:
        print("Order: no session-order adjustment or additional outlier exclusion; correlating raw pairs.")

    stats_formats = _load_stats_config(args.stats_config)

    res = correlate_psis(
        pairs_for_corr,
        output_dir=args.out,
        filename_stem=args.filename_stem,
        equal_axes=args.equal_axes,
        fixed_limits=tuple(args.fixed_limits) if args.fixed_limits else None,
        exclude_outliers=False,
        outlier_method=args.outlier_method,
        iqr_multiplier=args.iqr_multiplier,
        z_thresh=args.z_thresh,
        show_point_labels=args.show_labels,
        point_labels=filtered_labels,
        stats_formats=stats_formats,
    )
    stats_lines = res.get("stats_lines", [])
    tests_line = res.get("tests_line")
    if stats_lines:
        for line in stats_lines:
            print(line)
    if tests_line:
        print(tests_line)

    print(f"Wrote {FIGURE_NAME} correlation figure to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()

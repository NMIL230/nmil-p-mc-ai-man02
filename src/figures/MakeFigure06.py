#!/usr/bin/env python3
"""Figure 06: Aggregate RMSE evolution per sampling method.

Reads the results pickle produced by `src/pipelines/compare_generator_rmses.py`
and renders an aggregate mean±SD RMSE-over-samples figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Any, List

import numpy as np

# Ensure repo root is on sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.modeling.experiment2 import (
    load_results,
    determine_methods,
    method_label,
    pid_order_like_adaptive_scatter,
    plot_rmse_aggregate_single,
    summarize_rmse_comparison,
)
from src.utils.data.pids import collapse_repeat_sessions
from src.utils.reporting.formatting import format_stat
from src.utils.plotting.figure_naming import figure_output_dir, figure_output_stem

FIGURE_KEY = "Fig06"
FIGURE_STEM = figure_output_stem(FIGURE_KEY)
DEFAULT_FIGURE_OUTPUT = figure_output_dir(FIGURE_KEY)
FIGURE_NAME = DEFAULT_FIGURE_OUTPUT.name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Figure 06 (RMSE evolution) from results pickle")
    p.add_argument(
        "--input",
        type=Path,
        default=Path(f"{PROJECT_ROOT}/data/compare_generator_rmses.pkl"),
        help="Results pickle produced by compare_generator_rmses.py",
    )
    p.add_argument("--all-methods", action="store_true", help="Include Random and Active, Entropy Maximizing")
    p.add_argument("--exclude-repeats", action="store_true", help="Exclude subjects with _R# suffix (e.g., AMLEC_018_R1)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--reject-outliers", dest="reject_outliers", action="store_true", help="Reject outlier participants (default)")
    g.add_argument("--keep-outliers", dest="reject_outliers", action="store_false", help="Keep all participants (no outlier rejection)")
    p.set_defaults(reject_outliers=True)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (defaults next to input as *_rmse_aggregate_6.4in.png)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = load_results(args.input)
    participants: Dict[str, Any] = data.get("participants", {})
    if not participants:
        raise ValueError("No participants found in results file")

    methods: List[str] = determine_methods(participants, include_all=args.all_methods)
    labels = {m: method_label(m, include_all=args.all_methods) for m in methods}

    pid_order = pid_order_like_adaptive_scatter(participants, reject_outliers=bool(args.reject_outliers))
    if args.exclude_repeats:
        pid_order = collapse_repeat_sessions(pid_order)

    participants = {pid: participants[pid] for pid in pid_order if pid in participants}

    if args.out is None:
        out_png = DEFAULT_FIGURE_OUTPUT / f"{FIGURE_STEM}.png"
    else:
        out_png = args.out

    plot_rmse_aggregate_single(
        participants,
        methods,
        labels,
        out_png,
        width_in=6.4,
        height_in=4.2,
    )
    print(f"Wrote {FIGURE_NAME} to {out_png}")

    try:
        stats_point, stats_equiv = summarize_rmse_comparison(
            participants,
            independent_method="sequential_adaptive",
            active_method="max_entropy_heuristic",
            sample_index=29,
            equivalence_bf_threshold=3.0,
        )
        mean_bf10_series = np.asarray(stats_equiv.mean_bf10_series, dtype=float)
        std_bf10_series = np.asarray(stats_equiv.std_bf10_series, dtype=float)
        print(f"\n{FIGURE_STEM} Active − Independent Staircase @30 samples:")
        print(f"  n={stats_point.n}")
        print(
            "  Mean RMSE: Active="
            f"{format_stat(stats_point.mean_active, mode='decimal')} vs "
            f"Independent={format_stat(stats_point.mean_independent, mode='decimal')}"
        )
        print(
            "  Mean difference (Active−Independent)="
            f"{format_stat(stats_point.mean_diff, mode='decimal')} "
            f"(SD={format_stat(stats_point.mean_std, mode='decimal')}, "
            f"95% CI [{format_stat(stats_point.mean_ci_low, mode='decimal')}, "
            f"{format_stat(stats_point.mean_ci_high, mode='decimal')}])"
        )
        df_t = stats_point.n - 1 if stats_point.n else float("nan")
        print(
            f"  Paired t({df_t})={format_stat(stats_point.t_stat, mode='decimal')}, "
            f"p={format_stat(stats_point.p_value, mode='scientific')}, "
            f"BF10={format_stat(stats_point.bf10_mean, mode='scientific')}, "
            f"dz={format_stat(stats_point.cohen_dz, mode='decimal')}"
        )
        print(
            "  Std difference (Active−Independent)="
            f"{format_stat(stats_point.std_diff, mode='decimal')} "
            f"(ratio={format_stat(stats_point.std_ratio, mode='decimal')})"
        )
        df_pm = stats_point.n - 2 if stats_point.n else float("nan")
        print(
            f"  Pitman-Morgan t({df_pm})={format_stat(stats_point.pitman_morgan_t, mode='decimal')}, "
            f"p={format_stat(stats_point.pitman_morgan_p, mode='scientific')}, "
            f"BF10={format_stat(stats_point.bf10_std, mode='scientific')}"
        )

        def _min_bf10(values: List[float]) -> float:
            if not values:
                return float("nan")
            arr = np.asarray(values, dtype=float)
            if arr.size == 0:
                return float("nan")
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return float("nan")
            return float(np.nanmin(finite))

        def _last_bf10(series: np.ndarray) -> tuple[float, int | None]:
            if series.size == 0:
                return float("nan"), None
            finite_idx = np.where(np.isfinite(series))[0]
            if finite_idx.size == 0:
                return float("nan"), None
            last_idx = int(finite_idx[-1])
            return float(series[last_idx]), last_idx

        mean_idx = stats_equiv.first_mean_equivalent_index
        if mean_idx is not None:
            bf10_mean = mean_bf10_series[mean_idx]
            print(
                "  BF10 equivalence on means (threshold "
                f"{format_stat(stats_equiv.bf_threshold, mode='decimal')}): reached at "
                f"sample {mean_idx + 1} with BF10={format_stat(bf10_mean, mode='scientific')}"
            )
        else:
            min_bf10_mean = _min_bf10(stats_equiv.mean_bf10_series)
            print(
                "  BF10 equivalence on means (threshold "
                f"{format_stat(stats_equiv.bf_threshold, mode='decimal')}): not reached; "
                f"min BF10={format_stat(min_bf10_mean, mode='scientific')}"
            )

        std_idx = stats_equiv.first_std_equivalent_index
        if std_idx is not None:
            bf10_std = std_bf10_series[std_idx]
            print(
                "  BF10 equivalence on stds (threshold "
                f"{format_stat(stats_equiv.bf_threshold, mode='decimal')}): reached at "
                f"sample {std_idx + 1} with BF10={format_stat(bf10_std, mode='scientific')}"
            )
        else:
            print(
                "  BF10 equivalence on stds (threshold "
                f"{format_stat(stats_equiv.bf_threshold, mode='decimal')}): not reached; "
                f"min BF10={format_stat(_min_bf10(stats_equiv.std_bf10_series), mode='scientific')}"
            )

        last_mean_bf10, last_mean_idx = _last_bf10(mean_bf10_series)
        if last_mean_idx is not None:
            print(
                "  Final mean BF10 (sample "
                f"{last_mean_idx + 1})={format_stat(last_mean_bf10, mode='scientific')}"
            )
        else:
            print("  Final mean BF10 unavailable")

        last_std_bf10, last_std_idx = _last_bf10(std_bf10_series)
        if last_std_idx is not None:
            print(
                "  Final std BF10 (sample "
                f"{last_std_idx + 1})={format_stat(last_std_bf10, mode='scientific')}"
            )
        else:
            print("  Final std BF10 unavailable")
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"{FIGURE_STEM} statistical summary failed: {exc}")



if __name__ == "__main__":  # pragma: no cover
    main()

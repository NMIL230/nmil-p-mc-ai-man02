#!/usr/bin/env python3
"""Figure 07: All-methods 50% contours at 30 samples with 30–70% CI and GT.

Reads the results pickle produced by `src/pipelines/compare_generator_rmses.py`
and renders a small-multiples grid with one panel per participant, overlaying
all methods' 50% contour lines and shaded 30–70% credible region, plus the
ground-truth contour.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Any, List

# Ensure repo root is on sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.modeling.experiment2 import (
    load_results,
    determine_methods,
    method_label,
    pid_order_like_adaptive_scatter,
    plot_all_methods_overlay_30_ci,
)
from src.utils.data.pids import collapse_repeat_sessions
from src.utils.plotting.figure_naming import figure_output_dir, figure_output_stem

FIGURE_KEY = "Fig07"
FIGURE_STEM = figure_output_stem(FIGURE_KEY)
DEFAULT_FIGURE_OUTPUT = figure_output_dir(FIGURE_KEY)
FIGURE_NAME = DEFAULT_FIGURE_OUTPUT.name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Figure 07 overlays from results pickle")
    p.add_argument(
        "--input",
        type=Path,
        default=Path(f"{PROJECT_ROOT}/data/compare_generator_rmses.pkl"),
        help="Results pickle produced by compare_generator_rmses.py",
    )
    p.add_argument("--rows", type=int, default=5, help="Rows for subplot grid")
    p.add_argument("--cols", type=int, default=6, help="Cols for subplot grid")
    p.add_argument("--all-methods", action="store_true", help="Include Random and Active, Entropy Maximizing in plots")
    p.add_argument("--exclude-repeats", action="store_true", help="Exclude subjects with _R# suffix (e.g., AMLEC_018_R1)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--reject-outliers", dest="reject_outliers", action="store_true", help="Reject outlier participants (default)")
    g.add_argument("--keep-outliers", dest="reject_outliers", action="store_false", help="Keep all participants (no outlier rejection)")
    p.set_defaults(reject_outliers=True)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (defaults next to input under *_all_methods_overlays/all_methods_30_ci.png)",
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

    plot_all_methods_overlay_30_ci(participants, methods, labels, out_png, args.rows, args.cols)
    print(f"Wrote {FIGURE_NAME} to {out_png}")


if __name__ == "__main__":  # pragma: no cover
    main()

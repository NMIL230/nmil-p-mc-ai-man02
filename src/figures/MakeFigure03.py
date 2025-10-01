#!/usr/bin/env python3
"""Adaptive outcome scatter with 50% posterior boundary overlay."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Sequence

# Ensure repo root on sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data.common_inputs import build_fits_and_pairs
from src.utils.modeling.adaptive import (
    build_norm_cdf_generators,
    order_models_by_posterior_slope,
    plot_adaptive_scatter_with_posterior50,
)
from src.utils.statistics.outliers import filter_pids_by_pair_diffs
from src.utils.reporting.reporting import validity_removed_pids, format_validity_removed_line
from src.utils.reporting.reporting import format_validity_details
from src.utils.plotting.plotting import (
    CORR_OUTLIER_METHOD_DEFAULT,
    CORR_IQR_MULTIPLIER_DEFAULT,
    CORR_ZSCORE_THRESHOLD_DEFAULT,
)
from src.utils.plotting.constants import PSI_THETA_TEX
from src.utils.reporting.formatting import format_stat
from src.utils.plotting.figure_naming import figure_output_dir, figure_output_stem

FIGURE_KEY = "Fig03"
FIGURE_STEM = figure_output_stem(FIGURE_KEY)
DEFAULT_FIGURE_OUTPUT = figure_output_dir(FIGURE_KEY)
FIGURE_NAME = DEFAULT_FIGURE_OUTPUT.name

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate adaptive scatter + posterior50 figure")
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_FIGURE_OUTPUT,
        help="Output directory",
    )
    p.add_argument("--pickle", type=Path, default=None, help="AMLEC pickle path (optional)")
    # Validity mask control (mask)
    p.add_argument("--apply-validity-mask", dest="apply_validity_mask", action="store_true", help="Apply validity mask (default ON)")
    p.add_argument("--no-apply-validity-mask", dest="apply_validity_mask", action="store_false", help="Do not apply validity mask")
    p.set_defaults(apply_validity_mask=True)
    # Outlier handling (entity-level filter based on modality diffs)
    p.add_argument("--no-exclude-outliers", dest="exclude_outliers", action="store_false", help="Keep all subjects; disable outlier exclusion")
    p.add_argument("--exclude-outliers", dest="exclude_outliers", action="store_true", help="Enable outlier exclusion (default ON)")
    p.set_defaults(exclude_outliers=True)
    p.add_argument("--outlier-method", type=str, default=CORR_OUTLIER_METHOD_DEFAULT, choices=["iqr_diff", "zscore_diff", "none"], help="Outlier detection method for Classic vs Adaptive diffs")
    p.add_argument("--iqr-multiplier", type=float, default=CORR_IQR_MULTIPLIER_DEFAULT, help="IQR multiplier (for iqr_diff)")
    p.add_argument("--z-thresh", type=float, default=CORR_ZSCORE_THRESHOLD_DEFAULT, help="Z-score threshold (for zscore_diff)")
    p.add_argument("--sort-mode", type=str, choices=["avg", "target", "classic", "adaptive-3k-threshold"], default="classic", help="Entity ordering metric (avg, target slope, classic psi, or θ at K=3)")
    p.add_argument("--show-sort-metric", dest="show_sort_metric", action="store_true", help="Annotate each subplot with the chosen sort metric")
    p.add_argument("--no-show-sort-metric", dest="show_sort_metric", action="store_false", help="Disable subplot sort metric annotations (default)")
    p.set_defaults(show_sort_metric=False)
    p.add_argument("--show-classic-threshold", dest="show_classic_threshold", action="store_true", help=f"Plot classic {PSI_THETA_TEX} threshold marker (default ON)")
    p.add_argument("--no-show-classic-threshold", dest="show_classic_threshold", action="store_false", help=f"Hide classic {PSI_THETA_TEX} threshold marker")
    p.set_defaults(show_classic_threshold=True)
    p.add_argument("--nrows", type=int, default=None, help="Explicit number of subplot rows")
    p.add_argument("--ncols", type=int, default=None, help="Explicit number of subplot columns (default auto)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    fx, fy, ent_pairs, _pairs, valid_ent_pairs, _labels, gp_by_pid, adaptive_raw_points_all = build_fits_and_pairs(args.pickle)
    # Choose source for PID filtering based on validity mask flag
    if args.apply_validity_mask:
        removed = validity_removed_pids(ent_pairs, valid_ent_pairs)
        line = format_validity_removed_line(removed)
        if line:
            print(line)
            details = format_validity_details(ent_pairs, min_allowed=0.0, max_allowed=18.0)
            if details:
                print(details)
        valid_pids = {pid for (pid, _x, _y) in valid_ent_pairs}
        print("Order: applied validity mask; proceeding to optional outlier exclusion.")
        pairs_for_outliers = valid_ent_pairs
    else:
        valid_pids = set(gp_by_pid.keys()) | set(adaptive_raw_points_all.keys())
        pairs_for_outliers = ent_pairs
        print("Order: validity mask disabled; proceeding to optional outlier exclusion.")
    # Optional outlier exclusion at entity level using modality diffs
    if args.exclude_outliers and len(pairs_for_outliers) >= 3:
        kept_pids, info = filter_pids_by_pair_diffs(
            pairs_for_outliers,
            method=str(args.outlier_method),
            iqr_multiplier=float(args.iqr_multiplier),
            z_thresh=float(args.z_thresh),
        )
        if info.get("removed", 0) > 0:
            b = info.get("bounds")
            bounds_txt = "bounds=None"
            if b:
                lo_fmt = format_stat(b[0], mode="decimal")
                hi_fmt = format_stat(b[1], mode="decimal")
                bounds_txt = f"bounds={lo_fmt}..{hi_fmt}"
            print(f"Applied outlier exclusion on modality diffs ({info['method']}): removed {info['removed']}/{info['before']} ({bounds_txt}); kept {info['after']}")
        valid_pids = valid_pids & kept_pids if kept_pids else valid_pids
        print("Order: completed optional outlier exclusion; proceeding to adaptive scatter plotting on filtered set.")
    else:
        print("Order: no outlier exclusion requested; proceeding to adaptive scatter plotting.")
    # Filter adaptive points and GP by PID
    gp_by_pid = {pid: gp_by_pid[pid] for pid in gp_by_pid if pid in valid_pids}
    adaptive_raw_points_all = {pid: adaptive_raw_points_all[pid] for pid in adaptive_raw_points_all if pid in valid_pids}
    classic_for_models = {pid: fx[pid] for pid in fx if pid in valid_pids}
    classic_scores: Dict[str, float] = {}
    for pid, fit_list in classic_for_models.items():
        if not fit_list:
            continue
        try:
            classic_scores[pid] = float(fit_list[0].get("psi_theta", float("nan")))
        except (TypeError, ValueError):
            continue
    models = build_norm_cdf_generators(gp_by_pid, classic_for_models)
    entity_order, slopes, avg_slopes, theta_at_k = order_models_by_posterior_slope(
        models,
        target_k=3.0,
        sort_mode=args.sort_mode,
        classic_scores=classic_scores,
    )
    if entity_order:
        def _fmt(pid: str) -> str:
            slope = slopes.get(pid, float("nan"))
            avg = avg_slopes.get(pid, float("nan"))
            classic = classic_scores.get(pid, float("nan"))
            theta = theta_at_k.get(pid, float("nan"))
            tgt_txt = f"target={format_stat(slope, mode='decimal')}" if math.isfinite(slope) else "target=nan"
            avg_txt = f"avg={format_stat(avg, mode='decimal')}" if math.isfinite(avg) else "avg=nan"
            classic_txt = f"classic={format_stat(classic, mode='decimal')}" if math.isfinite(classic) else "classic=nan"
            theta_txt = f"theta@3={format_stat(theta, mode='decimal')}" if math.isfinite(theta) else "theta@3=nan"
            return f"{pid}({tgt_txt}, {avg_txt}, {classic_txt}, {theta_txt})"

        slope_desc = ", ".join(_fmt(pid) for pid in entity_order)
        print(f"Order (sort={args.sort_mode}, K=3): {slope_desc}")
    all_entities = set(gp_by_pid.keys()) | set(adaptive_raw_points_all.keys())
    if args.sort_mode == "avg":
        metric_map = {pid: avg_slopes.get(pid, float("nan")) for pid in all_entities}
        metric_label = "avg slope"
    elif args.sort_mode == "target":
        metric_map = {pid: slopes.get(pid, float("nan")) for pid in all_entities}
        metric_label = "target slope"
    elif args.sort_mode == "classic":
        metric_map = {pid: classic_scores.get(pid, float("nan")) for pid in all_entities}
        metric_label = "classic psi"
    else:
        metric_map = {pid: theta_at_k.get(pid, float("nan")) for pid in all_entities}
        metric_label = "theta@K=3"
    stem = FIGURE_STEM if args.sort_mode == "classic" else f"{FIGURE_STEM}_sort-{args.sort_mode}"
    plot_adaptive_scatter_with_posterior50(
        adaptive_raw_points_all,
        gp_by_pid,
        output_dir=args.out,
        filename_stem=stem,
        entity_order=entity_order,
        sort_metric=metric_map,
        sort_metric_name=metric_label,
        show_sort_metric=args.show_sort_metric,
        classic_thresholds=classic_scores,
        show_classic_threshold=args.show_classic_threshold,
        nrows_override=args.nrows,
        ncols_override=args.ncols,
    )
    print(f"Wrote {FIGURE_NAME} adaptive scatter + posterior50 figure to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()

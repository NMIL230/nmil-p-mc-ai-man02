#!/usr/bin/env python3
"""Adaptive probability surface (Norm-CDF) per entity."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence, Dict, Any

# Ensure repo root on sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data.common_inputs import build_fits_and_pairs
from src.utils.modeling.adaptive import (
    build_norm_cdf_generators,
    order_models_by_posterior_slope,
    plot_adaptive_probability_surface,
)
from src.utils.statistics.outliers import filter_pids_by_pair_diffs
from src.utils.reporting.reporting import validity_removed_pids, format_validity_removed_line
from src.utils.reporting.reporting import format_validity_details
from src.utils.plotting.plotting import (
    CORR_OUTLIER_METHOD_DEFAULT,
    CORR_IQR_MULTIPLIER_DEFAULT,
    CORR_ZSCORE_THRESHOLD_DEFAULT,
)
from src.utils.reporting.formatting import format_stat


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate adaptive probability surface figure")
    p.add_argument("--out", type=Path, default=Path("results/adaptive_probability"), help="Output directory")
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
    p.add_argument("--show-sort-metric", dest="show_sort_metric", action="store_true", help="Annotate each subplot with the chosen sort metric (default ON)")
    p.add_argument("--no-show-sort-metric", dest="show_sort_metric", action="store_false", help="Disable subplot sort metric annotations")
    p.set_defaults(show_sort_metric=True)
    p.add_argument("--no-use-gp-surface", dest="use_gp_surface", action="store_false", help="Render Norm-CDF model surface instead of GP posterior")
    p.add_argument("--use-gp-surface", dest="use_gp_surface", action="store_true", help="Render GP posterior surface (default)")
    p.set_defaults(use_gp_surface=True)
    p.add_argument("--nrows", type=int, default=None, help="Explicit number of subplot rows")
    p.add_argument("--ncols", type=int, default=None, help="Explicit number of subplot columns")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    fits_x, fits_y, ent_pairs, _pairs, valid_ent_pairs, _labels, gp_by_pid, _pts = build_fits_and_pairs(args.pickle)
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
        valid_pids = set(gp_by_pid.keys())
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
        print("Order: completed optional outlier exclusion; proceeding to probability surface plotting on filtered set.")
    else:
        print("Order: no outlier exclusion requested; proceeding to probability surface plotting.")
    gp_by_pid = {pid: gp_by_pid[pid] for pid in gp_by_pid if pid in valid_pids}
    fits_x = {pid: fits_x[pid] for pid in fits_x if pid in valid_pids}
    diag: Dict[str, Any] = {}
    generators = build_norm_cdf_generators(gp_by_pid, fits_x, diagnostics=diag)
    classic_scores: Dict[str, float] = {}
    for pid, fit_list in fits_x.items():
        if not fit_list:
            continue
        try:
            classic_scores[pid] = float(fit_list[0].get("psi_theta", float("nan")))
        except (TypeError, ValueError):
            continue
    entity_order, slopes, avg_slopes, theta_at_k = order_models_by_posterior_slope(
        generators,
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
    all_entities = set(generators.keys())
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
    plot_adaptive_probability_surface(
        generators,
        output_dir=args.out,
        filename_stem=f"adaptive_prob_surface_sort-{args.sort_mode}",
        diagnostics=diag,
        entity_order=entity_order,
        sort_metric=metric_map,
        sort_metric_name=metric_label,
        show_sort_metric=args.show_sort_metric,
        nrows_override=args.nrows,
        ncols_override=args.ncols,
        use_gp_surface=args.use_gp_surface,
    )
    print(f"Wrote adaptive probability surface figure to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()

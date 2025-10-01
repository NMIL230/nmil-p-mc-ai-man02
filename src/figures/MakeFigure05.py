#!/usr/bin/env python3
r"""Practice gains (Second − First $\psi_{\theta}$) by starting mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

# Ensure repo root on sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plotting.constants import PSI_THETA_TEX
from src.utils.data.parity import (
    build_parity_records,
    parity_dataframe,
    plot_delta_distributions,
    estimate_global_session_effect,
    apply_session_effect_adjustment,
)
from src.utils.data.common_inputs import build_fits_and_pairs
from src.utils.reporting.reporting import validity_removed_pids, format_validity_removed_line
from src.utils.reporting.reporting import format_validity_details
from src.utils.plotting.plotting import (
    CORR_OUTLIER_METHOD_DEFAULT,
    CORR_IQR_MULTIPLIER_DEFAULT,
    CORR_ZSCORE_THRESHOLD_DEFAULT,
)
from src.utils.reporting.formatting import format_stat
from src.utils.data.pids import normalize_pid
from src.utils.plotting.figure_naming import figure_output_dir

FIGURE_KEY = "Fig05"
DEFAULT_FIGURE_OUTPUT = figure_output_dir(FIGURE_KEY)
FIGURE_NAME = DEFAULT_FIGURE_OUTPUT.name


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate practice gains violin plot")
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_FIGURE_OUTPUT,
        help="Output directory",
    )
    # Outlier handling (defaults to exclusion on; global method)
    p.add_argument("--no-exclude-outliers", dest="exclude_outliers", action="store_false", help="Keep all subjects; disable outlier exclusion")
    p.add_argument("--exclude-outliers", dest="exclude_outliers", action="store_true", help="Enable outlier exclusion (default)")
    p.set_defaults(exclude_outliers=True)
    p.add_argument("--outlier-method", type=str, default=CORR_OUTLIER_METHOD_DEFAULT, choices=["iqr_diff", "zscore_diff", "none"], help="Outlier detection method for practice gains")
    p.add_argument("--iqr-multiplier", type=float, default=CORR_IQR_MULTIPLIER_DEFAULT, help="IQR multiplier (for iqr_diff)")
    p.add_argument("--z-thresh", type=float, default=CORR_ZSCORE_THRESHOLD_DEFAULT, help="Z-score threshold (for zscore_diff)")
    # Optional global session order adjustment and repeat handling
    p.add_argument("--adjust-session-effect", action="store_true", help="Regress out global second-vs-first session effect before plotting")
    p.add_argument("--exclude-repeats", action="store_true", help="Exclude participants with _R# repeat suffix")
    p.add_argument(
        "--stats-config",
        type=str,
        default=None,
        help="JSON array (or path) of format strings for per-group statistics.",
    )
    return p.parse_args(argv)


def _load_stats_config(raw: str | None) -> tuple[list[str] | None, list[str] | None]:
    if not raw:
        return None, None
    candidate = Path(raw)
    if candidate.exists():
        text = candidate.read_text(encoding="utf-8")
    else:
        text = raw
    config = json.loads(text)
    lines: list[str] | None = None
    order: list[str] | None = None
    if isinstance(config, dict):
        raw_lines = config.get("lines")
        if raw_lines is not None:
            if isinstance(raw_lines, str):
                lines = [raw_lines]
            else:
                if not isinstance(raw_lines, list):
                    raise ValueError("stats-config 'lines' must be a string or list of strings.")
                lines = [str(item) for item in raw_lines]
        raw_order = config.get("order")
        if raw_order is not None:
            if isinstance(raw_order, str):
                order = [raw_order]
            else:
                if not isinstance(raw_order, list):
                    raise ValueError("stats-config 'order' must be a string or list of strings.")
                order = [str(item) for item in raw_order]
        if lines is None and order is None:
            raise ValueError("stats-config dict must include 'lines' and/or 'order'.")
        return lines, order
    if isinstance(config, str):
        return [config], None
    if isinstance(config, list):
        return [str(item) for item in config], None
    raise ValueError("stats-config must be a JSON array, string, or object with 'lines'.")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    records = build_parity_records()
    df = parity_dataframe(records)
    # Apply validity mask to align with other figures
    try:
        _fx, _fy, ent_pairs, _pairs, valid_ent_pairs, _labels, _gp, _pts = build_fits_and_pairs()
        removed = validity_removed_pids(ent_pairs, valid_ent_pairs)
        line = format_validity_removed_line(removed)
        if line:
            print(line)
            details = format_validity_details(ent_pairs, min_allowed=0.0, max_allowed=18.0)
            if details:
                print(details)
        valid_pids = {pid for (pid, _x, _y) in valid_ent_pairs}
        before = len(df)
        df = df[df["pid"].isin(valid_pids)].copy()
        after = len(df)
        if before != after:
            print(f"Applied validity mask: removed {before - after}/{before} subjects; kept {after}")
    except Exception:
        pass
    df_filtered = df

    did_session_adjust = False
    if args.adjust_session_effect and len(df_filtered) >= 3:
        beta = estimate_global_session_effect(df_filtered)
        if abs(beta) > 0:
            beta_str = format_stat(beta, mode="decimal")
            if beta >= 0:
                beta_str = f"+{beta_str}"
            print(f"Estimated global session effect (second vs first): {beta_str}; removing from second-session {PSI_THETA_TEX}")
        df_filtered = apply_session_effect_adjustment(df_filtered, beta)
        did_session_adjust = True

    if did_session_adjust:
        print("Order: applied session-order adjustment prior to delta-distribution plotting.")
    else:
        print("Order: no session-order adjustment; plotting raw values.")

    stats_formats, stats_order = _load_stats_config(args.stats_config)

    plot_delta_distributions(
        df_filtered,
        args.out,
        enforce_validity=True,
        validity_min=0.0,
        validity_max=18.0,
        exclude_outliers=args.exclude_outliers,
        outlier_method=args.outlier_method,
        iqr_multiplier=float(args.iqr_multiplier),
        z_thresh=float(args.z_thresh),
        collapse_repeats=args.exclude_repeats,
        stats_formats=stats_formats,
        stats_order=stats_order,
    )
    print(f"Wrote {FIGURE_NAME} to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()

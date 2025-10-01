#!/usr/bin/env python3
"""Survey Data and Performance analysis aligned with Figure 04 participants."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plotting.plotting import DEFAULT_FIGURE_DPI

from src.analysis.extract_survey_subset import (
    DEFAULT_PICKLE_PATH,
    load_survey_payload,
)
from src.data.amlec_loader import load_amlec_diffmaps
from src.utils.data.common_inputs import build_fits_and_pairs
from src.utils.modeling.fits import fit_sigmoids_per_entity
from src.utils.data.pids import base_pid, normalize_pid, collapse_repeat_sessions
from src.utils.reporting.formatting import format_stat
from src.utils.reporting.reporting import format_ci_interval, format_df_value
from src.utils.statistics.bayes import compute_jzs_bf_corr


DEFAULT_OUTPUT_DIR = Path("manuscript_analysis") / "Survey Data and Performance"
DEFAULT_FIGURE_BASENAME = "Survey Data and Performance"
EQUIVALENCE_BOUND_DEFAULT = 0.5


@dataclass(frozen=True)
class SurveyMetricConfig:
    record_key: str
    axis_label: str
    description: str


SURVEY_METRICS: Dict[str, SurveyMetricConfig] = {
    "minecraft_familiarity": SurveyMetricConfig(
        record_key="familiarity",
        axis_label="Minecraft familiarity (self-reported, Q3.1)",
        description="Self-reported familiarity with Minecraft (Qualtrics Q3.1)",
    ),
    "mouse_keyboard_comfort": SurveyMetricConfig(
        record_key="mouse_keyboard_comfort",
        axis_label="Mouse & keyboard gaming comfort (self-reported)",
        description="Self-reported comfort using mouse and keyboard for gaming",
    ),
}

_NUMERIC_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)")


def _coerce_numeric(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except (TypeError, ValueError):
            match = _NUMERIC_PATTERN.search(text)
            if not match:
                return None
            try:
                numeric = float(match.group(1))
            except (TypeError, ValueError):
                return None
    if not np.isfinite(numeric):
        return None
    return float(numeric)


def _load_survey_metric_map(
    records: Sequence[Dict[str, str]],
    record_key: str,
) -> tuple[Dict[str, float], int, int]:
    value_map: Dict[str, float] = {}
    total_responses = 0
    skipped_non_numeric = 0
    for rec in records:
        pid = base_pid(rec.get("pid"))
        if not pid:
            continue
        raw_value = rec.get(record_key)
        if raw_value in (None, ""):
            continue
        total_responses += 1
        numeric = _coerce_numeric(raw_value)
        if numeric is None:
            skipped_non_numeric += 1
            continue
        if pid not in value_map:
            value_map[pid] = numeric
    return value_map, total_responses, skipped_non_numeric


def _fig04_filtered_participants(
    pickle_path: Path | None,
    *,
    exclude_repeats: bool,
    exclude_outliers: bool,
    outlier_method: str,
    iqr_multiplier: float,
    z_thresh: float,
) -> tuple[set[str], set[str], int, int]:
    _fx, _fy, ent_pairs, _pairs, valid_ent_pairs, _labels, _gp, _pts = build_fits_and_pairs(pickle_path)
    ent_source = list(valid_ent_pairs)

    if not ent_source:
        return set(), set(), len(valid_ent_pairs), 0

    pairs_array = np.array([(float(x), float(y)) for (_pid, x, y) in ent_source], dtype=float)
    kept_indices = np.arange(len(ent_source))

    if exclude_outliers and len(ent_source) >= 3:
        x_vals = pairs_array[:, 0]
        y_vals = pairs_array[:, 1]
        method = outlier_method
        mask = np.ones(len(ent_source), dtype=bool)
        if method == "iqr_diff":
            diffs = y_vals - x_vals
            q1, q3 = np.percentile(diffs, [25, 75])
            iqr = q3 - q1
            lo = q1 - iqr_multiplier * iqr
            hi = q3 + iqr_multiplier * iqr
            mask = (diffs >= lo) & (diffs <= hi)
        elif method == "iqr":
            def _bounds(arr: np.ndarray) -> tuple[float, float]:
                q1, q3 = np.percentile(arr, [25, 75])
                iqr = q3 - q1
                return (q1 - iqr_multiplier * iqr), (q3 + iqr_multiplier * iqr)

            x_lo, x_hi = _bounds(x_vals)
            y_lo, y_hi = _bounds(y_vals)
            mask = (x_vals >= x_lo) & (x_vals <= x_hi) & (y_vals >= y_lo) & (y_vals <= y_hi)
        elif method == "zscore_diff":
            diffs = y_vals - x_vals
            sd = diffs.std(ddof=1) if len(diffs) > 1 else diffs.std()
            mean = diffs.mean()
            if np.isfinite(sd) and sd > 0:
                z = (diffs - mean) / sd
            else:
                z = np.zeros_like(diffs)
            mask = np.abs(z) <= z_thresh
        elif method == "zscore":
            def _z(arr: np.ndarray) -> np.ndarray:
                sd = arr.std(ddof=1) if len(arr) > 1 else arr.std()
                if np.isfinite(sd) and sd > 0:
                    return (arr - arr.mean()) / sd
                return np.zeros_like(arr)

            zx = _z(x_vals)
            zy = _z(y_vals)
            mask = (np.abs(zx) <= z_thresh) & (np.abs(zy) <= z_thresh)
        kept_indices = np.where(mask)[0]

    filtered_ent_pairs = [ent_source[i] for i in kept_indices]
    if exclude_repeats and filtered_ent_pairs:
        filtered_pids = [pid for (pid, _x, _y) in filtered_ent_pairs]
        collapsed_pids = collapse_repeat_sessions(filtered_pids)
        selected_positions: List[int] = []
        used_positions: set[int] = set()
        for pid in collapsed_pids:
            for pos, value in enumerate(filtered_pids):
                if pos in used_positions:
                    continue
                if value == pid:
                    selected_positions.append(pos)
                    used_positions.add(pos)
                    break
        selected_positions.sort()
        filtered_ent_pairs = [filtered_ent_pairs[i] for i in selected_positions]

    allowed_full = {pid for (pid, _x, _y) in filtered_ent_pairs}
    allowed_base: set[str] = set()
    for pid in allowed_full:
        root = base_pid(pid)
        if root:
            allowed_base.add(root)
    return allowed_full, allowed_base, len(valid_ent_pairs), len(filtered_ent_pairs)


def _load_average_psi(
    pickle_path: Path | None,
    allowed_full_pids: set[str] | None = None,
) -> Dict[str, float]:
    if pickle_path is None:
        classic_maps, _adaptive_maps, _diag, _scatter, _pts, _gp = load_amlec_diffmaps()
    else:
        classic_maps, _adaptive_maps, _diag, _scatter, _pts, _gp = load_amlec_diffmaps(pickle_path)
    classic_fits = fit_sigmoids_per_entity(classic_maps, phantom_failure_at=17.0)
    values: Dict[str, List[float]] = defaultdict(list)
    for pid, fits in classic_fits.items():
        if allowed_full_pids is not None and pid not in allowed_full_pids:
            continue
        root = base_pid(pid)
        if not root:
            continue
        for fit in fits:
            psi = fit.get("psi_theta")
            try:
                numeric = float(psi)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                values[root].append(numeric)
    return {pid: float(np.mean(vals)) for pid, vals in values.items() if vals}


def _fisher_confidence_interval(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if not np.isfinite(r) or n <= 3:
        return float("nan"), float("nan")
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    z_crit = scipy_stats.norm.ppf(1 - alpha / 2)
    se = 1.0 / np.sqrt(max(n - 3, 1))
    lower = np.tanh(z - z_crit * se)
    upper = np.tanh(z + z_crit * se)
    return float(lower), float(upper)


def _correlation_power(n: int, rho: float, alpha: float = 0.05) -> float:
    if n <= 3 or not np.isfinite(rho) or abs(rho) >= 1.0:
        return float("nan")
    df = n - 2
    noncentrality = rho * np.sqrt(df / max(1.0 - rho**2, 1e-12))
    tcrit = scipy_stats.t.ppf(1 - alpha / 2, df)
    # Probability of failing to reject under H1 (type II)
    beta = scipy_stats.nct.cdf(tcrit, df, noncentrality) - scipy_stats.nct.cdf(-tcrit, df, noncentrality)
    power = 1.0 - beta
    return float(np.clip(power, 0.0, 1.0))


def _correlation_tost(
    r: float,
    n: int,
    equivalence_bound: float,
    alpha: float = 0.05,
) -> tuple[float, float, float, float, bool]:
    bound = abs(equivalence_bound)
    if n <= 3 or not np.isfinite(r):
        return float("nan"), float("nan"), False
    if bound <= 0 or bound >= 1:
        return float("nan"), float("nan"), False
    z_r = np.arctanh(np.clip(r, -0.999999, 0.999999))
    z_lower = np.arctanh(-bound)
    z_upper = np.arctanh(bound)
    scale = np.sqrt(max(n - 3, 1))
    stat_lower = (z_r - z_lower) * scale
    stat_upper = (z_r - z_upper) * scale
    p_lower = 1.0 - scipy_stats.norm.cdf(stat_lower)
    p_upper = scipy_stats.norm.cdf(stat_upper)
    reject = (p_lower < alpha) and (p_upper < alpha)
    return (
        float(stat_lower),
        float(stat_upper),
        float(p_lower),
        float(p_upper),
        bool(reject),
    )


def _plot_scatter(
    predictor: Dict[str, float],
    performance: Dict[str, float],
    output_path: Path,
    *,
    equivalence_bound: float,
    axis_label: str,
    metric_id: str,
    allowed_base_pids: set[str] | None = None,
) -> Dict[str, float]:
    if allowed_base_pids is not None:
        common = sorted(pid for pid in performance if pid in predictor and pid in allowed_base_pids)
    else:
        common = sorted(pid for pid in performance if pid in predictor)
    if not common:
        raise ValueError("No overlapping participants between survey responses and performance data")
    x = np.array([predictor[pid] for pid in common], dtype=float)
    y = np.array([performance[pid] for pid in common], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, color="#1f77b4", s=50)

    for pid, xi, yi in zip(common, x, y):
        ax.annotate(pid.replace("AMLEC_", ""), (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)

    stats_payload: Dict[str, float] = {}

    if x.size >= 3 and np.ptp(x) > 0 and np.std(y) > 0:
        coeffs = np.polyfit(x, y, deg=1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), linestyle="--", color="#444444", linewidth=1)

        r, p_value = scipy_stats.pearsonr(x, y)
        df = len(x) - 2
        if df > 0 and abs(r) < 1:
            denom = max(1e-12, 1 - r**2)
            t_stat = float(r * np.sqrt(df / denom))
        else:
            t_stat = float("nan")
        ci_low, ci_high = _fisher_confidence_interval(r, len(x))
        test_power = _correlation_power(len(x), equivalence_bound)
        z_lower, z_upper, p_lower, p_upper, tost_reject = _correlation_tost(r, len(x), equivalence_bound)
        ci90_low, ci90_high = _fisher_confidence_interval(r, len(x), alpha=0.10)

        slope, intercept = coeffs
        y_hat = slope * x + intercept
        sse_alt = float(np.sum((y - y_hat) ** 2))
        sse_null = float(np.sum((y - np.mean(y)) ** 2))
        n_obs = len(x)
        eps = 1e-12
        sse_alt = max(sse_alt, eps)
        sse_null = max(sse_null, eps)
        bic_null = n_obs * np.log(sse_null / n_obs) + np.log(n_obs)
        bic_alt = n_obs * np.log(sse_alt / n_obs) + 2 * np.log(n_obs)
        bf10_method = "JZS"
        try:
            bf10 = float(compute_jzs_bf_corr(x, y))
        except Exception:
            bf10_method = "BIC"
            bf10 = float(np.exp((bic_null - bic_alt) / 2.0))

        r2_val = float(r**2) if np.isfinite(r) else float("nan")

        stats_payload = {
            "n": float(len(x)),
            "r": float(r),
            "p_value": float(p_value),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "df": float(df) if df >= 0 else float("nan"),
            "t_stat": float(t_stat),
            "test_name": "Paired-sample Pearson correlation (two-tailed)",
            "power_at_bound": float(test_power),
            "tost_stat_lower": float(z_lower),
            "tost_stat_upper": float(z_upper),
            "tost_p_lower": float(p_lower),
            "tost_p_upper": float(p_upper),
            "tost_reject": float(tost_reject),
            "tost_ci90_low": float(ci90_low),
            "tost_ci90_high": float(ci90_high),
            "bf10": float(bf10),
            "bf10_method": bf10_method,
            "r2": float(r2_val),
            "allowed_base_ids": float(len(allowed_base_pids)) if allowed_base_pids is not None else float(len(x)),
        }

        r_fmt = format_stat(r, mode="decimal")
        p_fmt = format_stat(p_value, mode="scientific")
        t_fmt = format_stat(t_stat, mode="decimal")
        df_text = "NA" if df <= 0 else str(df)
        ci_text = format_ci_interval(ci_low, ci_high)
        power_fmt = format_stat(test_power, mode="decimal")
        bound_fmt = format_stat(equivalence_bound, mode="decimal")
        bf_fmt = format_stat(bf10, mode="scientific")
        r2_fmt = format_stat(r2_val, mode="decimal") if np.isfinite(r2_val) else "NA"

        annotation_lines = [
            f"Pearson correlation (paired): r = {r_fmt}; {ci_text}; R² = {r2_fmt}",
            f"t({df_text}) = {t_fmt}; p = {p_fmt}; n = {len(x)}",
            f"Power(|r|≥{bound_fmt}) = {power_fmt}",
            f"BF10 ≈ {bf_fmt}",
        ]
        tost_line = (
            "TOST: equivalence" if tost_reject else "TOST: inconclusive"
        ) + f" (n = {len(x)})"
        annotation_lines.append(tost_line)
        ax.text(
            0.02,
            0.98,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
        )

    ax.set_xlabel(axis_label)
    ax.set_ylabel(r"Average classic $\psi_{\theta}$ (lower is better)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_FIGURE_DPI)
    plt.close(fig)
    if stats_payload:
        stats_payload.setdefault("metric_id", metric_id)
        stats_payload.setdefault("axis_label", axis_label)
    return stats_payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot classic performance versus a survey metric")
    parser.add_argument(
        "--survey-csv",
        type=Path,
        default=None,
        help="Survey CSV export (defaults to latest Qualtrics export)",
    )
    parser.add_argument(
        "--survey-pickle",
        type=Path,
        default=None,
        help="Cached survey pickle (defaults to data/survey/survey_subset.pkl)",
    )
    parser.add_argument(
        "--pickle",
        type=Path,
        default=None,
        help="AMLEC dataset pickle (defaults to environment configuration)",
    )
    parser.add_argument(
        "--survey-metric",
        type=str,
        nargs="+",
        default=["minecraft_familiarity", "mouse_keyboard_comfort"],
        choices=sorted(list(SURVEY_METRICS.keys()) + ["all"]),
        help=(
            "Survey field(s) to correlate with performance. Provide one or more values; "
            "use 'all' to run every supported metric (default runs Minecraft familiarity and mouse/keyboard comfort)."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output figure path (defaults to manuscript_analysis/Survey Data and Performance/*.png)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for output figures when multiple survey metrics are requested (defaults to manuscript_analysis/Survey Data and Performance).",
    )
    parser.add_argument(
        "--equivalence-bound",
        type=float,
        default=EQUIVALENCE_BOUND_DEFAULT,
        help="Equivalence bound for |r| in TOST and power calculation (default: %(default)s)",
    )
    parser.add_argument(
        "--04-exclude-repeats",
        dest="fig04_exclude_repeats",
        action="store_true",
        help="Match Fig 04 by removing participants with _R# repeat suffix before alignment",
    )
    parser.add_argument(
        "--04-exclude-outliers",
        dest="fig04_exclude_outliers",
        action="store_true",
        help="Match Fig 04 default by excluding outliers when determining alignment (default)",
    )
    parser.add_argument(
        "--04-no-exclude-outliers",
        dest="fig04_exclude_outliers",
        action="store_false",
        help="Disable Fig 04 outlier exclusion during alignment",
    )
    parser.set_defaults(fig04_exclude_outliers=True)
    parser.add_argument(
        "--04-outlier-method",
        dest="fig04_outlier_method",
        choices=["iqr_diff", "iqr", "zscore_diff", "zscore"],
        default="iqr_diff",
        help="Outlier detection method mirroring Fig 04 alignment",
    )
    parser.add_argument(
        "--04-iqr-multiplier",
        dest="fig04_iqr_multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier for Fig 04 alignment (used when method is iqr_diff or iqr)",
    )
    parser.add_argument(
        "--04-z-thresh",
        dest="fig04_z_thresh",
        type=float,
        default=3.0,
        help="Z-score threshold for Fig 04 alignment (used when method is zscore/zscore_diff)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    refresh = args.survey_csv is not None
    survey_pickle = args.survey_pickle or DEFAULT_PICKLE_PATH
    payload = load_survey_payload(
        csv_path=args.survey_csv,
        pickle_path=survey_pickle,
        refresh=refresh,
    )
    records = payload.get("subset_records")
    if not isinstance(records, list) or not records:
        raise ValueError("Survey subset payload is empty; regenerate the cache from the survey CSV")
    source_csv = payload.get("source_csv")
    if source_csv:
        print(f"Loaded survey subset from {source_csv}")
    requested_metrics = [str(item) for item in args.survey_metric]
    metric_ids: List[str] = []
    for metric in requested_metrics:
        if metric == "all":
            metric_ids.extend(SURVEY_METRICS.keys())
        else:
            metric_ids.append(metric)
    # Deduplicate while preserving order
    seen: set[str] = set()
    metric_ids = [m for m in metric_ids if not (m in seen or seen.add(m))]

    if not metric_ids:
        raise ValueError("No survey metrics requested")

    unknown = [m for m in metric_ids if m not in SURVEY_METRICS]
    if unknown:
        raise ValueError(f"Unknown survey metric(s): {', '.join(sorted(set(unknown)))}")

    if len(metric_ids) > 1 and args.out is not None:
        raise ValueError(
            "--out cannot be combined with multiple survey metrics; use --out-dir instead."
        )

    allowed_full_pids, allowed_base_pids, fig04_total_valid, fig04_filtered = _fig04_filtered_participants(
        args.pickle,
        exclude_repeats=args.fig04_exclude_repeats,
        exclude_outliers=args.fig04_exclude_outliers,
        outlier_method=args.fig04_outlier_method,
        iqr_multiplier=float(args.fig04_iqr_multiplier),
        z_thresh=float(args.fig04_z_thresh),
    )
    if not allowed_full_pids:
        print("Fig 04 alignment produced no participants; aborting analysis.")
        return
    print(
        f"Fig 04 alignment: retained {fig04_filtered}/{fig04_total_valid} participants after validity/outlier filtering "
        f"(unique base IDs {len(allowed_base_pids)})."
    )

    performance_map = _load_average_psi(args.pickle, allowed_full_pids)
    performance_map = {
        pid: value
        for pid, value in performance_map.items()
        if pid in allowed_base_pids
    }
    missing_perf = len(allowed_base_pids - set(performance_map.keys()))
    print(
        f"Performance data available for {len(performance_map)}/{len(allowed_base_pids)} Fig 04-aligned participants"
        + (f" (missing {missing_perf})" if missing_perf else "")
        + "."
    )
    if not performance_map:
        print("No performance data available after Fig 04 alignment; aborting analysis.")
        return
    equivalence_bound = float(args.equivalence_bound)
    any_results = False

    for metric_id in metric_ids:
        metric_cfg = SURVEY_METRICS[metric_id]
        predictor_map, total_responses, skipped_non_numeric = _load_survey_metric_map(
            records,
            metric_cfg.record_key,
        )
        if not predictor_map:
            print(
                f"Skipping metric '{metric_id}': no numeric responses found in cached survey records."
            )
            continue

        pre_align_count = len(predictor_map)
        predictor_map = {
            pid: value for pid, value in predictor_map.items() if pid in allowed_base_pids
        }
        removed_for_alignment = pre_align_count - len(predictor_map)
        if removed_for_alignment > 0:
            print(
                f"Excluded {removed_for_alignment} numeric responses for metric '{metric_id}' not present in the Fig 04 subset."
            )
        if not predictor_map:
            print(
                f"Skipping metric '{metric_id}': no overlapping respondents with Fig 04 subset."
            )
            continue

        if args.out is not None:
            output_path = args.out
        else:
            output_dir = args.out_dir or DEFAULT_OUTPUT_DIR
            safe_label = re.sub(r"[^\w\-]+", " ", metric_cfg.axis_label).strip()
            if len(metric_ids) == 1:
                filename = f"{DEFAULT_FIGURE_BASENAME}.png"
            else:
                filename = f"{DEFAULT_FIGURE_BASENAME} - {safe_label}.png"
            output_path = output_dir / filename

        stats_payload = _plot_scatter(
            predictor_map,
            performance_map,
            output_path,
            equivalence_bound=equivalence_bound,
            axis_label=metric_cfg.axis_label,
            metric_id=metric_id,
            allowed_base_pids=allowed_base_pids,
        )
        stats_payload.setdefault("metric_id", metric_id)
        stats_payload.setdefault("axis_label", metric_cfg.axis_label)
        stats_payload["responses_total"] = int(total_responses)
        stats_payload["responses_before_fig04_filter"] = int(pre_align_count)
        stats_payload["responses_parsed"] = int(len(predictor_map))
        stats_payload["responses_removed_not_in_fig04"] = int(removed_for_alignment)
        stats_payload["responses_skipped_non_numeric"] = int(skipped_non_numeric)
        stats_payload["equivalence_bound"] = equivalence_bound

        print(
            fr"Wrote {metric_cfg.axis_label} vs classic $\psi_{{\theta}}$ figure to {output_path}"
        )
        if total_responses:
            print(
                f"Parsed {len(predictor_map)}/{total_responses} numeric survey responses for metric '{metric_id}'."
            )
        if skipped_non_numeric:
            print(
                f"Skipped {skipped_non_numeric} additional responses for metric '{metric_id}' because they were not numeric."
            )
        if stats_payload:
            n = int(stats_payload.get("n", 0))
            r = stats_payload.get("r")
            p_value = stats_payload.get("p_value")
            ci_low = stats_payload.get("ci_low")
            ci_high = stats_payload.get("ci_high")
            df_val = stats_payload.get("df")
            t_val = stats_payload.get("t_stat")
            test_name = stats_payload.get("test_name", "Pearson correlation")
            power_at_bound = stats_payload.get("power_at_bound")
            tost_stat_lower = stats_payload.get("tost_stat_lower")
            tost_stat_upper = stats_payload.get("tost_stat_upper")
            tost_lower = stats_payload.get("tost_p_lower")
            tost_upper = stats_payload.get("tost_p_upper")
            tost_reject = bool(stats_payload.get("tost_reject"))
            tost_ci90_low = stats_payload.get("tost_ci90_low")
            tost_ci90_high = stats_payload.get("tost_ci90_high")
            bf10 = stats_payload.get("bf10")
            bf_method = stats_payload.get("bf10_method")
            r2_val = stats_payload.get("r2")
            axis_label = stats_payload.get("axis_label", metric_cfg.axis_label)
            r_str = format_stat(r, mode="decimal")
            p_str = format_stat(p_value, mode="scientific")
            t_str = format_stat(t_val, mode="decimal")
            df_str = format_df_value(df_val)
            power_str = format_stat(power_at_bound, mode="decimal")
            bound_str = format_stat(equivalence_bound, mode="decimal")
            ci_text = format_ci_interval(ci_low, ci_high)
            r2_text = format_stat(r2_val, mode="decimal") if np.isfinite(r2_val) else "NA"
            print(
                fr"{test_name} ({axis_label} vs classic $\psi_{{\theta}}$): r={r_str}; {ci_text}; R²={r2_text}; t({df_str})={t_str}; p={p_str}; n={n}"
            )
            print(
                f"Power to detect |r| ≥ {bound_str}: {power_str}"
            )
            if bf10 is not None and np.isfinite(bf10):
                direction = "favoring H₁" if bf10 > 1 else "favoring H₀"
                print(
                    f"Bayes factor BF10 ≈ {format_stat(bf10, mode='scientific')} ({direction})"
                )
            if np.isfinite(tost_lower) and np.isfinite(tost_upper):
                tost_status = "passes" if tost_reject else "fails"
                p_lower_str = format_stat(tost_lower, mode="scientific")
                p_upper_str = format_stat(tost_upper, mode="scientific")
                z_lower_str = format_stat(tost_stat_lower, mode="decimal")
                z_upper_str = format_stat(tost_stat_upper, mode="decimal")
                ci90_low_str = format_stat(tost_ci90_low, mode="decimal")
                ci90_high_str = format_stat(tost_ci90_high, mode="decimal")
                print(
                    "TOST equivalence test (bound ±{}): z_lower={}, z_upper={}, df=∞, p_lower={}, p_upper={}, 90% CI [{}, {}], n={} → {} equivalence"
                    .format(bound_str, z_lower_str, z_upper_str, p_lower_str, p_upper_str, ci90_low_str, ci90_high_str, n, tost_status)
                )
            print(
                "Tests: Pearson correlation t-test (two-tailed, Fisher z CI); "
                "Bayes factor BF10; TOST equivalence (two one-sided z-tests)."
            )
        else:
            print(
                "Not enough spread in survey metric or performance to evaluate correlation statistics."
            )
        any_results = True

    if not any_results:
        print("No figures were generated; see messages above for details.")


if __name__ == "__main__":  # pragma: no cover
    main()

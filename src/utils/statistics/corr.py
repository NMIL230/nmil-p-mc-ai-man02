#!/usr/bin/env python3
"""Correlation helpers shared by manuscript figures and analyses."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Ensure workspace root available when executed directly (e.g., VS Code "Run")
if __name__ == "__main__":  # pragma: no cover
    import sys

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.utils.plotting.plotting import (
    AXIS_BUFFER_RATIO,
    AXIS_LABEL_FONT_SIZE,
    CORR_EQUAL_AXES_DEFAULT,
    CORR_FIXED_LIMITS_DEFAULT,
    CORR_JITTER_CLOSE_THRESH,
    CORR_JITTER_MAX_JITTER,
    CORR_LABEL_CLASH_THRESHOLD,
    CORR_LABEL_PLACEMENT_ORDER,
    CORR_PID_LABEL_SUFFIX_DIGITS,
    CORR_SHOW_PID_LABELS_DEFAULT,
    CORR_DEFAULT_FILENAME_STEM,
    CORR_DEFAULT_TITLE,
    CORR_DEFAULT_X_LABEL,
    CORR_DEFAULT_Y_LABEL,
    DECIMALS_FOR_TICKS,
    FIG_SIZE,
    MARKER_ALPHA,
    MARKER_COLOR,
    MARKER_SIZE,
    N_MAJOR_TICKS,
    REG_COLOR,
    REG_FILL_ALPHA,
    REG_LINE_WIDTH,
    SUPER_TITLE_FONT_SIZE,
    DEFAULT_FIGURE_DPI,
    _style_spines_and_ticks,
)
from src.utils.reporting.formatting import format_stat
from src.utils.reporting.reporting import format_ci_interval, format_df_pair, format_df_value
from src.utils.statistics.bayes import compute_jzs_bf_corr


@dataclass
class PreparedData:
    x: np.ndarray
    y: np.ndarray
    labels: Optional[np.ndarray]


@dataclass
class FilterResult:
    x: np.ndarray
    y: np.ndarray
    labels: Optional[np.ndarray]
    keep_indices: np.ndarray
    bounds: Optional[Tuple[float, float]]


@dataclass
class RegressionResult:
    model: Optional[sm.regression.linear_model.RegressionResultsWrapper]
    r_squared: float


@dataclass
class CorrelationMetrics:
    pearson_r: float
    pearson_p: float
    pearson_df: float
    pearson_t: float
    pearson_ci_low: float
    pearson_ci_high: float
    spearman_r: float
    spearman_p: float
    spearman_df: float
    spearman_t: float
    spearman_ci_low: float
    spearman_ci_high: float


@dataclass
class ICCMetrics:
    icc: float
    icc_ci_low: float
    icc_ci_high: float
    f_stat: float
    p_value: float
    df1: float
    df2: float


# ---------------------------------------------------------------------------
# Input preparation and filtering
# ---------------------------------------------------------------------------


def _prepare_arrays(
    pairs: Iterable[Tuple[float, float]],
    point_labels: Optional[Iterable[str]],
) -> Optional[PreparedData]:
    raw_pairs = np.asarray(list(pairs), dtype=float)
    if raw_pairs.size == 0:
        return None
    labels_arr = None
    if point_labels is not None:
        labels_arr = np.asarray(list(point_labels))
    return PreparedData(x=raw_pairs[:, 0], y=raw_pairs[:, 1], labels=labels_arr)


def _filter_outliers(
    data: PreparedData,
    *,
    exclude_outliers: bool,
    outlier_method: str,
    iqr_multiplier: float,
    z_thresh: float,
) -> FilterResult:
    keep_indices = np.arange(len(data.x))
    bounds: Optional[Tuple[float, float]] = None

    if exclude_outliers and len(data.x) > 0:
        if outlier_method == "iqr_diff":
            diffs = data.y - data.x
            q1, q3 = np.percentile(diffs, [25, 75])
            iqr = q3 - q1
            lo, hi = (q1 - iqr_multiplier * iqr), (q3 + iqr_multiplier * iqr)
            bounds = (float(lo), float(hi))
            mask = (diffs >= lo) & (diffs <= hi)
            keep_indices = np.where(mask)[0]
        elif outlier_method == "iqr":
            def _bounds(arr: np.ndarray) -> Tuple[float, float]:
                q1, q3 = np.percentile(arr, [25, 75])
                iqr = q3 - q1
                return (q1 - iqr_multiplier * iqr), (q3 + iqr_multiplier * iqr)

            x_lo, x_hi = _bounds(data.x)
            y_lo, y_hi = _bounds(data.y)
            mask = (data.x >= x_lo) & (data.x <= x_hi) & (data.y >= y_lo) & (data.y <= y_hi)
            keep_indices = np.where(mask)[0]
        elif outlier_method == "zscore_diff":
            diffs = data.y - data.x
            denom = diffs.std(ddof=1) if len(diffs) > 1 else diffs.std()
            denom = float(denom) if np.isfinite(denom) and denom is not None else 0.0
            if denom > 0:
                z = (diffs - diffs.mean()) / denom
                mask = np.abs(z) <= z_thresh
                mu = float(diffs.mean())
                bounds = (mu - z_thresh * denom, mu + z_thresh * denom)
            else:
                mask = np.ones(len(diffs), dtype=bool)
            keep_indices = np.where(mask)[0]
        elif outlier_method == "zscore":
            zx = (data.x - data.x.mean()) / (data.x.std(ddof=1) if len(data.x) > 1 else data.x.std())
            zy = (data.y - data.y.mean()) / (data.y.std(ddof=1) if len(data.y) > 1 else data.y.std())
            mask = (np.abs(zx) <= z_thresh) & (np.abs(zy) <= z_thresh)
            keep_indices = np.where(mask)[0]
        # else "none": retain original keep_indices

    filtered_x = data.x[keep_indices]
    filtered_y = data.y[keep_indices]
    filtered_labels = data.labels[keep_indices] if data.labels is not None else None

    return FilterResult(
        x=filtered_x,
        y=filtered_y,
        labels=filtered_labels,
        keep_indices=keep_indices,
        bounds=bounds,
    )


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------


def _fit_regression(x: np.ndarray, y: np.ndarray) -> RegressionResult:
    try:
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        r2 = float(model.rsquared)
    except Exception:
        model = None
        r2 = float("nan")
    return RegressionResult(model=model, r_squared=r2)


def _compute_bayes_factor(
    x: np.ndarray,
    y: np.ndarray,
    regression: RegressionResult,
) -> Tuple[float, Optional[str]]:
    bf10 = float("nan")
    method: Optional[str] = None

    try:
        bf10 = float(compute_jzs_bf_corr(x, y))
        method = "JZS"
    except Exception:
        bf10 = float("nan")

    if not np.isfinite(bf10) and regression.model is not None and len(x) >= 3:
        try:
            sse_alt = float(np.sum(regression.model.resid ** 2))
        except Exception:
            sse_alt = float(getattr(regression.model, "ssr", float("nan")))
        sse_null = float(np.sum((y - y.mean()) ** 2))
        eps = 1e-12
        if np.isfinite(sse_alt) and np.isfinite(sse_null):
            sse_alt = max(sse_alt, eps)
            sse_null = max(sse_null, eps)
            n_obs = len(x)
            bic_null = n_obs * math.log(sse_null / n_obs) + 1 * math.log(n_obs)
            bic_alt = n_obs * math.log(sse_alt / n_obs) + 2 * math.log(n_obs)
            bf10 = float(math.exp((bic_null - bic_alt) / 2.0))
            method = "BIC"
    return bf10, method


def _compute_correlation_metrics(x: np.ndarray, y: np.ndarray) -> CorrelationMetrics:
    pearson_r = pearson_p = pearson_df = pearson_t = float("nan")
    pearson_ci_low = pearson_ci_high = float("nan")
    spearman_r = spearman_p = spearman_df = spearman_t = float("nan")
    spearman_ci_low = spearman_ci_high = float("nan")

    if len(x) >= 2:
        pearson_r, pearson_p = stats.pearsonr(x, y)
        df_corr = len(x) - 2
        if df_corr > 0 and np.isfinite(pearson_r) and abs(pearson_r) < 1:
            pearson_df = float(df_corr)
            denom = max(1e-12, 1 - pearson_r**2)
            pearson_t = float(pearson_r * math.sqrt(df_corr / denom))
            pearson_ci_low, pearson_ci_high = _fisher_z_confidence_interval(pearson_r, len(x))

        spearman_res = stats.spearmanr(x, y, nan_policy="omit")
        spearman_r = float(spearman_res.correlation)
        spearman_p = float(spearman_res.pvalue)
        if df_corr > 0 and np.isfinite(spearman_r) and abs(spearman_r) < 1:
            spearman_df = float(df_corr)
            denom = max(1e-12, 1 - spearman_r**2)
            spearman_t = float(spearman_r * math.sqrt(df_corr / denom))
            spearman_ci_low, spearman_ci_high = _fisher_z_confidence_interval(spearman_r, len(x))

    return CorrelationMetrics(
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        pearson_df=pearson_df,
        pearson_t=pearson_t,
        pearson_ci_low=pearson_ci_low,
        pearson_ci_high=pearson_ci_high,
        spearman_r=spearman_r,
        spearman_p=spearman_p,
        spearman_df=spearman_df,
        spearman_t=spearman_t,
        spearman_ci_low=spearman_ci_low,
        spearman_ci_high=spearman_ci_high,
    )


def _compute_icc_metrics(x: np.ndarray, y: np.ndarray) -> ICCMetrics:
    if len(x) == 0:
        return ICCMetrics(
            icc=float("nan"),
            icc_ci_low=float("nan"),
            icc_ci_high=float("nan"),
            f_stat=float("nan"),
            p_value=float("nan"),
            df1=float("nan"),
            df2=float("nan"),
        )

    scores = np.column_stack((x, y))
    mean_subject = scores.mean(axis=1, keepdims=True)
    mean_rater = scores.mean(axis=0, keepdims=True)
    grand_mean = float(scores.mean())
    k = scores.shape[1]
    n_val = int(len(x))
    df_subjects = n_val - 1
    df_raters = k - 1
    df_error = df_subjects * df_raters

    if df_subjects <= 0 or df_error <= 0:
        return ICCMetrics(
            icc=float("nan"),
            icc_ci_low=float("nan"),
            icc_ci_high=float("nan"),
            f_stat=float("nan"),
            p_value=float("nan"),
            df1=float(df_subjects) if df_subjects > 0 else float("nan"),
            df2=float(df_error) if df_error > 0 else float("nan"),
        )

    ss_subjects = k * np.sum((mean_subject.squeeze() - grand_mean) ** 2)
    ss_raters = n_val * np.sum((mean_rater.squeeze() - grand_mean) ** 2)
    residual = scores - mean_subject - mean_rater + grand_mean
    ss_error = np.sum(residual ** 2)

    ms_subjects = ss_subjects / df_subjects if df_subjects > 0 else np.nan
    ms_raters = ss_raters / df_raters if df_raters > 0 else np.nan
    ms_error = ss_error / df_error if df_error > 0 else np.nan

    ratio_component = float("nan")
    if np.isfinite(ms_error) and ms_error > 0 and np.isfinite(ms_raters):
        ratio_component = (k * (ms_raters - ms_error)) / (n_val * ms_error)
    elif np.isfinite(ms_error) and ms_error > 0:
        ratio_component = 0.0

    denominator = ms_subjects + (k - 1) * ms_error
    if np.isfinite(ratio_component):
        denominator += ratio_component * ms_error
    icc_val = (ms_subjects - ms_error) / denominator if denominator != 0 else np.nan

    if not np.isfinite(ms_error) or ms_error <= 0:
        f_val = np.inf
        p_val = 0.0
        icc_ci_low = icc_ci_high = float("nan")
    else:
        f_val = ms_subjects / ms_error if ms_error != 0 else np.inf
        p_val = float(1 - stats.f.cdf(f_val, df_subjects, df_error))
        p_val = float(np.clip(p_val, 0.0, 1.0))
        alpha = 0.05
        try:
            f_crit_upper = stats.f.ppf(1 - alpha / 2, df_subjects, df_error)
            f_crit_lower = stats.f.ppf(alpha / 2, df_subjects, df_error)
            if (
                np.isfinite(f_crit_upper)
                and np.isfinite(f_crit_lower)
                and f_crit_upper > 0
                and f_crit_lower > 0
                and np.isfinite(ratio_component)
            ):
                ratio_term = ratio_component if np.isfinite(ratio_component) else 0.0
                lower_component = f_val / f_crit_upper
                upper_component = f_val / f_crit_lower
                icc_ci_low = (lower_component - 1) / (lower_component + (k - 1) + ratio_term)
                icc_ci_high = (upper_component - 1) / (upper_component + (k - 1) + ratio_term)
            else:
                icc_ci_low = icc_ci_high = float("nan")
        except Exception:
            icc_ci_low = icc_ci_high = float("nan")

    return ICCMetrics(
        icc_val,
        icc_ci_low,
        icc_ci_high,
        float(f_val),
        float(p_val),
        float(df_subjects),
        float(df_error),
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _jitter_overlapping_points(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    close_thresh = CORR_JITTER_CLOSE_THRESH
    max_jitter = CORR_JITTER_MAX_JITTER
    x_plot = x.copy()
    y_plot = y.copy()
    used = np.zeros(len(x), dtype=bool)

    for i in range(len(x)):
        if used[i]:
            continue
        cluster = [i]
        for j in range(i + 1, len(x)):
            if used[j]:
                continue
            if abs(x_plot[i] - x_plot[j]) < close_thresh and abs(y_plot[i] - y_plot[j]) < close_thresh:
                cluster.append(j)
        if len(cluster) > 1:
            for idx in cluster:
                used[idx] = True
                x_plot[idx] += random.uniform(-max_jitter, max_jitter)
                y_plot[idx] += random.uniform(-max_jitter, max_jitter)
        else:
            used[i] = True
    return x_plot, y_plot


def _calculate_axis_limits(
    x_plot: np.ndarray,
    y_plot: np.ndarray,
    equal_axes: Optional[bool],
    fixed_limits: Optional[Tuple[float, float]],
) -> Tuple[float, float, float, float]:
    if equal_axes is None:
        equal_axes = CORR_EQUAL_AXES_DEFAULT
    if fixed_limits is None:
        fixed_limits = CORR_FIXED_LIMITS_DEFAULT

    def round_down(value: float, decimals: int = DECIMALS_FOR_TICKS) -> float:
        factor = 10**decimals
        return math.floor(value * factor) / factor

    def round_up(value: float, decimals: int = DECIMALS_FOR_TICKS) -> float:
        factor = 10**decimals
        return math.ceil(value * factor) / factor

    if fixed_limits is not None:
        lo, hi = fixed_limits
        return lo, hi, lo, hi

    x_min, x_max = float(x_plot.min()), float(x_plot.max())
    y_min, y_max = float(y_plot.min()), float(y_plot.max())
    if math.isclose(x_min, x_max, rel_tol=1e-9):
        x_min -= 1e-6
        x_max += 1e-6
    if math.isclose(y_min, y_max, rel_tol=1e-9):
        y_min -= 1e-6
        y_max += 1e-6

    x_range = x_max - x_min
    y_range = y_max - y_min
    x_buffer = x_range * AXIS_BUFFER_RATIO
    y_buffer = y_range * AXIS_BUFFER_RATIO
    x_min_buf = round_down(x_min - x_buffer)
    x_max_buf = round_up(x_max + x_buffer)
    y_min_buf = round_down(y_min - y_buffer)
    y_max_buf = round_up(y_max + y_buffer)

    if equal_axes:
        lo = min(x_min_buf, y_min_buf)
        hi = max(x_max_buf, y_max_buf)
        return lo, hi, lo, hi
    return x_min_buf, x_max_buf, y_min_buf, y_max_buf


def _render_plot(
    filtered: FilterResult,
    regression: RegressionResult,
    *,
    output_dir: Optional[Path],
    title: str,
    x_label: str,
    y_label: str,
    filename_stem: str,
    equal_axes: Optional[bool],
    fixed_limits: Optional[Tuple[float, float]],
    show_point_labels: Optional[bool],
    stats_lines_for_fig: List[str],
) -> None:
    if output_dir is None:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_plot, y_plot = _jitter_overlapping_points(filtered.x, filtered.y)

    fig, ax = plt.subplots(1, 1, figsize=(FIG_SIZE, FIG_SIZE))
    ax.scatter(x_plot, y_plot, s=MARKER_SIZE, alpha=MARKER_ALPHA, color=MARKER_COLOR, zorder=3)

    if regression.model is not None:
        new_x = np.linspace(min(x_plot), max(x_plot), 100)
        new_X = sm.add_constant(new_x)
        try:
            prediction = regression.model.get_prediction(new_X)
            pred = prediction.summary_frame(alpha=0.05)
            ax.plot(new_x, pred["mean"], color=REG_COLOR, linewidth=REG_LINE_WIDTH, linestyle="-", zorder=2)
            ax.fill_between(new_x, pred["mean_ci_lower"], pred["mean_ci_upper"], color=REG_COLOR, alpha=REG_FILL_ALPHA, zorder=1)
        except Exception:
            pass

    x_min_buf, x_max_buf, y_min_buf, y_max_buf = _calculate_axis_limits(x_plot, y_plot, equal_axes, fixed_limits)
    ax.set_xlim(x_min_buf, x_max_buf)
    ax.set_ylim(y_min_buf, y_max_buf)

    xticks = np.linspace(x_min_buf, x_max_buf, N_MAJOR_TICKS)
    yticks = np.linspace(y_min_buf, y_max_buf, N_MAJOR_TICKS)
    ax.set_xticks([round(v, DECIMALS_FOR_TICKS) for v in xticks])
    ax.set_yticks([round(v, DECIMALS_FOR_TICKS) for v in yticks])

    ax.set_title(title, fontsize=SUPER_TITLE_FONT_SIZE)
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONT_SIZE)
    _style_spines_and_ticks(ax)

    if show_point_labels is None:
        show_point_labels = CORR_SHOW_PID_LABELS_DEFAULT
    if filtered.labels is not None and show_point_labels:
        font_size = max(6, AXIS_LABEL_FONT_SIZE - 4)
        placed_orients: List[Tuple[float, float, str]] = []
        proximity = float(CORR_LABEL_CLASH_THRESHOLD)
        for idx, label in enumerate(filtered.labels):
            text = str(label).strip()
            if not text:
                continue
            chosen = None
            for orient, (dx, dy) in CORR_LABEL_PLACEMENT_ORDER:
                conflict = any(
                    porient == orient and abs(px - x_plot[idx]) < proximity and abs(py - y_plot[idx]) < proximity
                    for px, py, porient in placed_orients
                )
                if conflict:
                    continue
                chosen = (orient, (dx, dy))
                break
            if chosen is None:
                orient, (dx, dy) = CORR_LABEL_PLACEMENT_ORDER[-1]
            else:
                orient, (dx, dy) = chosen
            placed_orients.append((x_plot[idx], y_plot[idx], orient))
            ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
            va = "bottom" if dy > 0 else ("top" if dy < 0 else "center")
            ax.annotate(
                text,
                (x_plot[idx], y_plot[idx]),
                textcoords="offset points",
                xytext=(dx, dy),
                ha=ha,
                va=va,
                fontsize=font_size,
                color="black",
                zorder=4,
                clip_on=True,
            )

    lims = ax.get_xlim()
    diag_lo = min(lims[0], ax.get_ylim()[0])
    diag_hi = max(lims[1], ax.get_ylim()[1])
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], color="black", linestyle="--", linewidth=1.5, alpha=0.8)

    stats_text = "\n".join(stats_lines_for_fig)
    ax.text(
        0.05,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=AXIS_LABEL_FONT_SIZE - 2,
    )

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{filename_stem}.png"
    pdf_path = output_dir / f"{filename_stem}.pdf"
    plt.savefig(png_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _fisher_z_confidence_interval(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if not np.isfinite(r) or n <= 3:
        return float("nan"), float("nan")
    r_clipped = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r_clipped)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    se = 1.0 / math.sqrt(max(n - 3, 1))
    lower = np.tanh(z - z_crit * se)
    upper = np.tanh(z + z_crit * se)
    return float(lower), float(upper)


def _build_stats_sections(
    *,
    icc: ICCMetrics,
    corr: CorrelationMetrics,
    bf10: float,
    bf_method: Optional[str],
    r_squared: float,
    sample_size: int,
    stats_formats: Optional[List[str]],
) -> Tuple[Dict[str, str], List[str], List[str], str, str]:
    n_text = str(sample_size)
    bf_value = format_stat(bf10, mode="scientific") if np.isfinite(bf10) else "NA"
    bf_line_default = f"BF10={bf_value}" if bf_value != "NA" else "BF10=NA"

    mapping = {
        "icc": format_stat(icc.icc, mode="decimal"),
        "icc_ci": format_ci_interval(icc.icc_ci_low, icc.icc_ci_high),
        "f_stat": format_stat(icc.f_stat, mode="decimal"),
        "icc_p": format_stat(icc.p_value, mode="scientific"),
        "n": n_text,
        "pearson_r": format_stat(corr.pearson_r, mode="decimal"),
        "pearson_ci": format_ci_interval(corr.pearson_ci_low, corr.pearson_ci_high),
        "pearson_t": format_stat(corr.pearson_t, mode="decimal"),
        "pearson_df": format_df_value(corr.pearson_df),
        "pearson_p": format_stat(corr.pearson_p, mode="scientific"),
        "r2": format_stat(r_squared, mode="decimal") if np.isfinite(r_squared) else "NA",
        "spearman_r": format_stat(corr.spearman_r, mode="decimal"),
        "spearman_ci": format_ci_interval(corr.spearman_ci_low, corr.spearman_ci_high),
        "spearman_t": format_stat(corr.spearman_t, mode="decimal"),
        "spearman_df": format_df_value(corr.spearman_df),
        "spearman_p": format_stat(corr.spearman_p, mode="scientific"),
        "bf10": bf_line_default,
        "bf10_value": bf_value,
        "bf10_method": bf_method or "",
    }

    if stats_formats:
        stats_lines = [str(fmt).format(**mapping) for fmt in stats_formats]
        stats_lines_for_fig = list(stats_lines)
    else:
        stats_lines = [
            f"ICC={mapping['icc']}",
            mapping["icc_ci"],
            f"F{format_df_pair(icc.df1, icc.df2)}={mapping['f_stat']}",
            f"p={mapping['icc_p']}",
            f"n={n_text}",
            "",
            f"Pearson r={mapping['pearson_r']}",
            mapping["pearson_ci"],
            f"R²={mapping['r2']}",
            f"t({mapping['pearson_df']})={mapping['pearson_t']}",
            f"p={mapping['pearson_p']}",
            f"n={n_text}",
        ]
        if np.isfinite(corr.spearman_r):
            stats_lines.extend([
                "",
                f"Spearman ρ={mapping['spearman_r']}",
                mapping["spearman_ci"],
                f"t({mapping['spearman_df']})={mapping['spearman_t']}",
                f"p={mapping['spearman_p']}",
                f"n={n_text}",
            ])
        stats_lines.append("")
        stats_lines.append(bf_line_default)
        stats_lines_for_fig = list(stats_lines)

    tests_parts = [
        "ICC F-test (two-way random absolute agreement)",
        "Pearson correlation t-test (two-tailed, Fisher z CI)",
    ]
    if np.isfinite(corr.spearman_r):
        tests_parts.append("Spearman correlation t-test (two-tailed, Fisher z CI)")
    tests_parts.append("Bayes factor BF10")
    tests_line = "Tests: " + "; ".join(tests_parts) + "."

    if not stats_formats:
        stats_lines_for_fig.append(tests_line)

    return mapping, stats_lines, stats_lines_for_fig, tests_line, bf_value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def correlate_psis(
    pairs: Iterable[Tuple[float, float]],
    title: str = CORR_DEFAULT_TITLE,
    x_label: str = CORR_DEFAULT_X_LABEL,
    y_label: str = CORR_DEFAULT_Y_LABEL,
    output_dir: Optional[Path] = None,
    filename_stem: str = CORR_DEFAULT_FILENAME_STEM,
    equal_axes: Optional[bool] = None,
    fixed_limits: Optional[Tuple[float, float]] = None,
    exclude_outliers: Optional[bool] = None,
    outlier_method: str = "iqr_diff",
    iqr_multiplier: float = 1.5,
    z_thresh: float = 3.0,
    point_labels: Optional[Iterable[str]] = None,
    show_point_labels: Optional[bool] = None,
    stats_formats: Optional[List[str]] = None,
) -> Dict[str, any]:
    prepared = _prepare_arrays(pairs, point_labels)
    if prepared is None:
        return {
            "icc": np.nan,
            "p": np.nan,
            "n": 0,
            "f": np.nan,
            "filtered_pairs": [],
            "filtered_labels": [],
        }

    if exclude_outliers is None:
        exclude_outliers = True

    filtered = _filter_outliers(
        prepared,
        exclude_outliers=exclude_outliers,
        outlier_method=outlier_method,
        iqr_multiplier=iqr_multiplier,
        z_thresh=z_thresh,
    )

    regression = _fit_regression(filtered.x, filtered.y)
    bf10, bf_method = _compute_bayes_factor(filtered.x, filtered.y, regression)
    corr_metrics = _compute_correlation_metrics(filtered.x, filtered.y)
    icc_metrics = _compute_icc_metrics(filtered.x, filtered.y)

    mapping, stats_lines, stats_lines_for_fig, tests_line, bf_value = _build_stats_sections(
        icc=icc_metrics,
        corr=corr_metrics,
        bf10=bf10,
        bf_method=bf_method,
        r_squared=regression.r_squared,
        sample_size=len(filtered.x),
        stats_formats=stats_formats,
    )

    _render_plot(
        filtered,
        regression,
        output_dir=output_dir,
        title=title,
        x_label=x_label,
        y_label=y_label,
        filename_stem=filename_stem,
        equal_axes=equal_axes,
        fixed_limits=fixed_limits,
        show_point_labels=show_point_labels,
        stats_lines_for_fig=stats_lines_for_fig,
    )

    return {
        "icc": icc_metrics.icc,
        "icc_ci_low": icc_metrics.icc_ci_low,
        "icc_ci_high": icc_metrics.icc_ci_high,
        "r2": regression.r_squared,
        "p": icc_metrics.p_value,
        "n": len(filtered.x),
        "f": icc_metrics.f_stat,
        "pearson_r": corr_metrics.pearson_r,
        "pearson_p": corr_metrics.pearson_p,
        "pearson_df": corr_metrics.pearson_df,
        "pearson_t": corr_metrics.pearson_t,
        "pearson_ci_low": corr_metrics.pearson_ci_low,
        "pearson_ci_high": corr_metrics.pearson_ci_high,
        "pearson_test_name": "Paired-sample Pearson correlation (two-tailed)",
        "spearman_r": corr_metrics.spearman_r,
        "spearman_p": corr_metrics.spearman_p,
        "spearman_df": corr_metrics.spearman_df,
        "spearman_t": corr_metrics.spearman_t,
        "spearman_ci_low": corr_metrics.spearman_ci_low,
        "spearman_ci_high": corr_metrics.spearman_ci_high,
        "spearman_test_name": "Paired-sample Spearman rank correlation (two-tailed)",
        "bf10": bf10,
        "bf10_method": bf_method,
        "filtered_pairs": list(zip(filtered.x, filtered.y)),
        "filtered_labels": list(filtered.labels) if filtered.labels is not None else [],
        "kept_indices": filtered.keep_indices,
        "outlier_bounds": filtered.bounds,
        "outlier_method": outlier_method,
        "icc_df1": icc_metrics.df1,
        "icc_df2": icc_metrics.df2,
        "icc_test_name": "ICC absolute-agreement F-test",
        "stats_lines": stats_lines_for_fig,
        "tests_line": tests_line,
        "bf10_value": bf_value,
    }

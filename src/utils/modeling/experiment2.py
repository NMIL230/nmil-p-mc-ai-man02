from __future__ import annotations

"""Shared helpers for Experiment 2 figures (07, 06).

This module mirrors a minimal subset of logic from
`src/pipelines/plot_generator_rmse_results.py` so figure scripts can remain
tiny and import their behavior without depending on the pipeline module.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats as scipy_stats

# Ensure repo root is on sys.path when executed as a file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data.common_inputs import build_fits_and_pairs
from src.utils.modeling.adaptive import build_norm_cdf_generators, order_models_by_posterior_slope
from src.utils.statistics.outliers import filter_pids_by_pair_diffs
from src.utils.reporting.reporting import validity_removed_pids, format_validity_removed_line, format_validity_details
from src.utils.plotting.plotting import (
    AXIS_LABEL_FONT_SIZE,
    SUPER_TITLE_FONT_SIZE,
    TIGHT_RECT,
    OVERLAY_PER_PANEL_SIZE,
    OVERLAY_PANEL_W_RATIO,
    OVERLAY_PANEL_H_RATIO,
    OVERLAY_SUBPLOT_HSPACE,
    OVERLAY_SUBPLOT_WSPACE,
    SUBPLOT_TICK_LABEL_FONT_SIZE_X,
    SUBPLOT_TICK_LABEL_FONT_SIZE_Y,
    SUBPLOT_TITLE_FONT_SIZE,
    DEFAULT_FIGURE_DPI,
    _style_spines_and_ticks,
)
from src.utils.plotting.small_multiples import (
    create_small_multiples_layout,
    AxisSpec,
    TitleSpec,
    apply_axis_spec,
    apply_titles_and_layout,
    hide_extra_axes,
)
from src.utils.statistics.bayes import compute_jzs_bf_one_sample, compute_jzs_bf_corr


DEFAULT_METHOD_ORDER = [
    "sequential_adaptive",
    "halton",
    "max_entropy_heuristic",
]


@dataclass
class RMSEPointSummary:
    sample_index: int
    sample_count: int
    n: int
    mean_diff: float
    mean_std: float
    mean_ci_low: float
    mean_ci_high: float
    t_stat: float
    p_value: float
    cohen_dz: float
    bf10_mean: float
    mean_active: float
    mean_independent: float
    std_active: float
    std_independent: float
    std_diff: float
    std_ratio: float
    pitman_morgan_t: float
    pitman_morgan_p: float
    bf10_std: float


@dataclass
class RMSEEquivalenceSummary:
    bf_threshold: float
    mean_bf10_series: List[float]
    std_bf10_series: List[float]
    first_mean_equivalent_index: Optional[int]
    first_std_equivalent_index: Optional[int]


def method_label(method: str, include_all: bool) -> str:
    if method == "sequential_adaptive":
        return "Independent Staircase"
    if method == "halton":
        return "Halton"
    if method == "random":
        return "Random"
    if method == "max_entropy_heuristic":
        return "Active, Heuristic Constraint" if include_all else "Active"
    if method == "max_entropy":
        return "Active, Entropy Maximizing"
    return str(method)


def load_results(path: Path) -> Dict[str, Any]:
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "participants" not in data:
        raise ValueError("Unexpected results file structure: missing 'participants'")
    return data


def determine_methods(participants: Dict[str, Any], include_all: bool) -> List[str]:
    methods: List[str] = []
    method_set = set()
    for _pid, pdata in participants.items():
        exps = pdata.get("experiments", {})
        for m in exps.keys():
            if m not in method_set:
                method_set.add(m)
                methods.append(m)
    ordered = [m for m in DEFAULT_METHOD_ORDER if m in method_set]
    ordered += [m for m in methods if m not in ordered]
    if not include_all:
        ordered = [m for m in ordered if m not in ("random", "max_entropy")]
    return ordered


def _axis_spec_default() -> AxisSpec:
    return AxisSpec(
        x_min=1.0,
        x_max=17.0,
        y_min=1.0,
        y_max=8.0,
        x_ticks=range(1, 18, 2),
        y_ticks=range(1, 9),
        x_buffer_frac=0.05,
        y_buffer_frac=0.05,
    )


def pid_order_like_adaptive_scatter(
    participants: Dict[str, Any],
    *,
    reject_outliers: bool = True,
) -> List[str]:
    fx, fy, ent_pairs, _pairs, valid_ent_pairs, _labels, gp_by_pid, adaptive_raw_points_all = build_fits_and_pairs(None)
    removed = validity_removed_pids(ent_pairs, valid_ent_pairs)
    if removed:
        line = format_validity_removed_line(removed)
        if line:
            print(line)
            details = format_validity_details(ent_pairs, min_allowed=0.0, max_allowed=18.0)
            if details:
                print(details)
    valid_pids = {pid for (pid, _x, _y) in valid_ent_pairs}
    if reject_outliers:
        kept_pids, _info = filter_pids_by_pair_diffs(valid_ent_pairs)
        if kept_pids:
            valid_pids = valid_pids & kept_pids

    result_pids = set(participants.keys())
    valid_pids = [pid for pid in valid_pids if pid in result_pids]

    classic_for_models = {pid: fx.get(pid, []) for pid in valid_pids}
    gp_by_pid_filtered = {pid: gp_by_pid[pid] for pid in gp_by_pid if pid in valid_pids}
    models = build_norm_cdf_generators(gp_by_pid_filtered, classic_for_models)
    classic_scores: Dict[str, float] = {}
    for pid, fit_list in classic_for_models.items():
        if not fit_list:
            continue
        try:
            classic_scores[pid] = float(fit_list[0].get("psi_theta", float("nan")))
        except (TypeError, ValueError):
            continue
    entity_order, _slopes, _avg_slopes, _theta_at_k = order_models_by_posterior_slope(
        models, target_k=3.0, sort_mode="classic", classic_scores=classic_scores
    )
    ordered = [pid for pid in entity_order if pid in result_pids]
    return ordered


def _render_contour_small_multiples(
    participants: Dict[str, Any],
    rows: int,
    cols: int,
    title_text: str,
    output_path: Path,
    panel_drawer,
    legend_handles: List[Line2D] | None = None,
) -> None:
    pids = list(participants.keys())
    n = len(pids)
    if rows * cols < n:
        rows = int(np.ceil(n / cols))

    axis_spec = _axis_spec_default()

    layout = create_small_multiples_layout(
        n,
        nrows_override=rows,
        ncols_override=cols,
        include_legend_slot=True,
        per_panel_size=OVERLAY_PER_PANEL_SIZE,
        panel_w_ratio=OVERLAY_PANEL_W_RATIO,
        panel_h_ratio=OVERLAY_PANEL_H_RATIO,
    )
    for ax in layout.panel_axes:
        apply_axis_spec(ax, axis_spec)
        _style_spines_and_ticks(ax)
        ax.grid(False)

    for idx, pid in enumerate(pids):
        ax = layout.panel_axes[idx]
        panel_drawer(ax, pid, participants[pid])
        ax.tick_params(axis='x', labelsize=SUBPLOT_TICK_LABEL_FONT_SIZE_X)
        ax.tick_params(axis='y', labelsize=SUBPLOT_TICK_LABEL_FONT_SIZE_Y)
        row, col = layout.positions[idx]
        if row != layout.bottom_row_for_col.get(col, row):
            ax.tick_params(labelbottom=False)
        if col != 0:
            ax.tick_params(labelleft=False)

    if layout.legend_ax is not None:
        layout.legend_ax.axis('off')
        if legend_handles:
            layout.legend_ax.legend(
                handles=legend_handles,
                labels=[h.get_label() for h in legend_handles],
                loc='center',
                frameon=False,
            )

    title_spec = TitleSpec(
        title=title_text,
        xlabel="Pattern Size (Blocks)",
        ylabel="Number of Unique Colors",
        title_fontsize=SUPER_TITLE_FONT_SIZE * 2.2,
        xlabel_fontsize=AXIS_LABEL_FONT_SIZE * 1.8,
        ylabel_fontsize=AXIS_LABEL_FONT_SIZE * 1.8,
        tight_layout_rect=(0.06, 0.08, 0.99, 0.96),
        subplot_wspace=OVERLAY_SUBPLOT_WSPACE,
        subplot_hspace=OVERLAY_SUBPLOT_HSPACE,
    )
    apply_titles_and_layout(layout, title_spec)
    hide_extra_axes(layout)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    layout.fig.savefig(output_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    layout.fig.savefig(output_path.with_suffix('.pdf'), dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    plt.close(layout.fig)


def _compute_50pct_contour(x_ticks: np.ndarray, y_ticks: np.ndarray, zz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_ticks = np.asarray(x_ticks, dtype=float).ravel()
    y_ticks = np.asarray(y_ticks, dtype=float).ravel()
    zz_arr = np.asarray(zz, dtype=float)
    if zz_arr.ndim != 2:
        zz_arr = np.asarray(zz, dtype=float)
    if zz_arr.shape != (y_ticks.size, x_ticks.size):
        if zz_arr.shape == (x_ticks.size, y_ticks.size):
            zz_arr = zz_arr.T
        else:
            try:
                zz_arr = zz_arr.reshape(y_ticks.size, x_ticks.size)
            except Exception:
                pass
    thetas: list[float] = []
    for row in zz_arr:
        vals = np.asarray(row, dtype=float).ravel()
        idx = np.where((vals[:-1] - 0.5) * (vals[1:] - 0.5) <= 0)[0]
        if idx.size == 0:
            thetas.append(np.nan)
            continue
        i = int(idx[0])
        x0 = float(x_ticks[i]); x1 = float(x_ticks[i + 1])
        y0 = float(vals[i]); y1 = float(vals[i + 1])
        if y1 == y0:
            theta = x0
        else:
            t = (0.5 - y0) / (y1 - y0)
            theta = x0 + t * (x1 - x0)
        thetas.append(float(theta))
    return y_ticks.astype(float), np.asarray(thetas, dtype=float)


def plot_all_methods_overlay_30_ci(
    participants: Dict[str, Any],
    methods: List[str],
    labels: Dict[str, str],
    out_png: Path,
    rows: int,
    cols: int,
) -> None:
    sns.set(style="white")
    palette = sns.color_palette("tab10", n_colors=len(methods))
    method_color = {m: palette[i % len(palette)] for i, m in enumerate(methods)}

    handles = [Line2D([0], [0], color=method_color[m], lw=2.0, label=labels.get(m, m)) for m in methods]
    handles.append(Line2D([0], [0], color='black', lw=2.0, linestyle='--', label='Ground Truth'))

    def panel_drawer(ax, pid: str, pdata: Dict[str, Any]) -> None:
        exps = pdata.get("experiments", {})
        for m in methods:
            fp = exps.get(m, {}).get("posterior_30", {})
            x_ticks = fp.get("x_ticks"); y_ticks = fp.get("y_ticks"); zz = fp.get("zz")
            if x_ticks is None or y_ticks is None or zz is None:
                continue
            color = method_color[m]
            X, Y = np.meshgrid(np.asarray(x_ticks, dtype=float), np.asarray(y_ticks, dtype=float))
            Z = np.asarray(zz, dtype=float)
            if Z.shape == (len(x_ticks), len(y_ticks)):
                Z = Z.T
            ax.contourf(X, Y, Z, levels=[0.3, 0.7], colors=[color], alpha=0.15)
            k_post, theta_post = _compute_50pct_contour(x_ticks, y_ticks, zz)
            mask = np.isfinite(k_post) & np.isfinite(theta_post)
            if np.any(mask):
                ax.plot(theta_post[mask], k_post[mask], color=color, linewidth=2.0, label=m, zorder=3)
        gt = pdata.get("ground_truth", {})
        k_gt = np.asarray(gt.get("k", []), dtype=float)
        theta_gt = np.asarray(gt.get("theta", []), dtype=float)
        if k_gt.size and theta_gt.size:
            ax.plot(theta_gt, k_gt, color="black", linestyle="--", linewidth=2.0, label="Ground Truth", zorder=10)
        ax.set_title(pid, fontsize=SUBPLOT_TITLE_FONT_SIZE)
        ax.set_xlim(1, 17)
        ax.set_ylim(1, 8)

    title = "Predicted 50% Contour for Distinct Sampling Procedures after 30 Samples"
    _render_contour_small_multiples(
        participants,
        rows,
        cols,
        title_text=title,
        output_path=out_png,
        panel_drawer=panel_drawer,
        legend_handles=handles,
    )


def _common_length(participants: Dict[str, Any], methods: List[str]) -> int:
    lengths: List[int] = []
    for _pid, pdata in participants.items():
        exps = pdata.get("experiments", {})
        for m in methods:
            rmse = exps.get(m, {}).get("rmse", [])
            if isinstance(rmse, list):
                lengths.append(len(rmse))
    return int(np.min(lengths)) if lengths else 0


def plot_rmse_aggregate_single(
    participants: Dict[str, Any],
    methods: List[str],
    labels: Dict[str, str],
    out_png: Path,
    *,
    width_in: float = 6.4,
    height_in: Optional[float] = None,
) -> None:
    min_len = _common_length(participants, methods)
    x = np.arange(min_len)
    sns.set(style="white")
    palette = sns.color_palette("tab10", n_colors=len(methods))
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(methods)}

    if height_in is None:
        height_in = width_in / 2.0
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    _style_spines_and_ticks(ax)
    ax.grid(False)
    title_fs = 12
    label_fs = 10
    tick_fs = 10
    margins = dict(left=0.07, right=0.995, top=0.92, bottom=0.14)
    ax.tick_params(axis='both', labelsize=tick_fs)
    for m in methods:
        ys: List[np.ndarray] = []
        for _pid, pdata in participants.items():
            rmse = pdata.get("experiments", {}).get(m, {}).get("rmse", [])
            if not rmse:
                continue
            ys.append(np.asarray(rmse[:min_len], dtype=float))
        if not ys:
            continue
        Y = np.vstack(ys)
        mean = np.nanmean(Y, axis=0)
        std = np.nanstd(Y, axis=0)
        ax.plot(x, mean, label=labels.get(m, m), color=color_map[m], linewidth=2.0)
        ax.fill_between(x, mean - std, mean + std, color=color_map[m], alpha=0.2)
    ax.set_xlabel("Sample Count", fontsize=label_fs, labelpad=2)
    ax.set_ylabel("RMSE", fontsize=label_fs, labelpad=0.5)
    ax.set_title("RMSE Evolution of Distinct Sampling Procedures", fontsize=title_fs, pad=4)
    ax.legend(frameon=False, fontsize=label_fs - 1, loc='upper right')
    fig.subplots_adjust(**margins)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DEFAULT_FIGURE_DPI)
    fig.savefig(out_png.with_suffix('.pdf'), dpi=DEFAULT_FIGURE_DPI)
    plt.close(fig)


def _collect_rmse_pairs(
    participants: Dict[str, Any],
    method_a: str,
    method_b: str,
) -> Tuple[np.ndarray, np.ndarray]:
    arrays_a: List[np.ndarray] = []
    arrays_b: List[np.ndarray] = []
    min_len: Optional[int] = None
    for _pid, pdata in participants.items():
        exps = pdata.get("experiments", {})
        rmse_a = exps.get(method_a, {}).get("rmse")
        rmse_b = exps.get(method_b, {}).get("rmse")
        if not rmse_a or not rmse_b:
            continue
        arr_a = np.asarray(rmse_a, dtype=float)
        arr_b = np.asarray(rmse_b, dtype=float)
        if arr_a.size == 0 or arr_b.size == 0:
            continue
        if min_len is None:
            min_len = min(arr_a.size, arr_b.size)
        else:
            min_len = min(min_len, arr_a.size, arr_b.size)
        arrays_a.append(arr_a)
        arrays_b.append(arr_b)
    if not arrays_a or min_len is None or min_len <= 0:
        return np.empty((0, 0)), np.empty((0, 0))
    stack_a = np.vstack([arr[:min_len] for arr in arrays_a])
    stack_b = np.vstack([arr[:min_len] for arr in arrays_b])
    return stack_a, stack_b


def _paired_variance_stats(
    independent: np.ndarray,
    active: np.ndarray,
) -> Tuple[float, float, float]:
    mask = np.isfinite(independent) & np.isfinite(active)
    if mask.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    vals_ind = independent[mask]
    vals_act = active[mask]
    sums = vals_act + vals_ind
    diffs = vals_act - vals_ind
    r = np.corrcoef(sums, diffs)[0, 1]
    if not np.isfinite(r):
        return float("nan"), float("nan"), float("nan")
    n = mask.sum()
    denom = max(1e-12, 1.0 - r ** 2)
    t_val = r * np.sqrt(n - 2) / np.sqrt(denom)
    p_val = 2.0 * (1.0 - scipy_stats.t.cdf(abs(t_val), df=n - 2))
    try:
        bf10 = float(compute_jzs_bf_corr(sums, diffs))
    except Exception:
        bf10 = float("nan")
    return float(t_val), float(p_val), bf10


def _rmse_point_summary(
    independent: np.ndarray,
    active: np.ndarray,
    sample_index: int,
) -> RMSEPointSummary:
    mask = np.isfinite(independent) & np.isfinite(active)
    vals_ind = independent[mask]
    vals_act = active[mask]
    n = vals_ind.size
    diff = vals_act - vals_ind
    mean_diff = float(np.mean(diff)) if n else float("nan")
    mean_std = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
    mean_ci_low = mean_ci_high = float("nan")
    t_stat = p_value = float("nan")
    cohen_dz = float("nan")
    bf10_mean = float("nan")
    if n > 1 and np.isfinite(mean_std) and mean_std > 0:
        se = mean_std / np.sqrt(n)
        t_stat = mean_diff / se
        p_value = 2.0 * (1.0 - scipy_stats.t.cdf(abs(t_stat), df=n - 1))
        t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
        mean_ci_low = mean_diff - t_crit * se
        mean_ci_high = mean_diff + t_crit * se
        cohen_dz = mean_diff / mean_std
        try:
            bf10_mean = float(compute_jzs_bf_one_sample(diff))
        except Exception:
            bf10_mean = float("nan")
    mean_active = float(np.mean(vals_act)) if n else float("nan")
    mean_independent = float(np.mean(vals_ind)) if n else float("nan")
    std_active = float(np.std(vals_act, ddof=1)) if n > 1 else float("nan")
    std_independent = float(np.std(vals_ind, ddof=1)) if n > 1 else float("nan")
    std_diff = (
        float(std_active - std_independent)
        if np.isfinite(std_active) and np.isfinite(std_independent)
        else float("nan")
    )
    std_ratio = (
        float(std_active / std_independent)
        if np.isfinite(std_active) and np.isfinite(std_independent) and std_independent != 0
        else float("nan")
    )
    pitman_t, pitman_p, bf10_std = _paired_variance_stats(independent, active)
    return RMSEPointSummary(
        sample_index=sample_index,
        sample_count=sample_index + 1,
        n=int(n),
        mean_diff=mean_diff,
        mean_std=mean_std,
        mean_ci_low=float(mean_ci_low),
        mean_ci_high=float(mean_ci_high),
        t_stat=float(t_stat),
        p_value=float(p_value),
        cohen_dz=float(cohen_dz),
        bf10_mean=float(bf10_mean),
        mean_active=mean_active,
        mean_independent=mean_independent,
        std_active=std_active,
        std_independent=std_independent,
        std_diff=std_diff,
        std_ratio=std_ratio,
        pitman_morgan_t=float(pitman_t),
        pitman_morgan_p=float(pitman_p),
        bf10_std=float(bf10_std),
    )


def _equivalence_summary(
    independent_matrix: np.ndarray,
    active_matrix: np.ndarray,
    *,
    bf_threshold: float,
) -> RMSEEquivalenceSummary:
    if independent_matrix.size == 0 or active_matrix.size == 0:
        return RMSEEquivalenceSummary(bf_threshold, [], [], None, None)
    num_samples = independent_matrix.shape[1]
    mean_bf10_series: List[float] = []
    std_bf10_series: List[float] = []
    first_mean_idx: Optional[int] = None
    first_std_idx: Optional[int] = None
    for idx in range(num_samples):
        ind = independent_matrix[:, idx]
        act = active_matrix[:, idx]
        mask = np.isfinite(ind) & np.isfinite(act)
        if idx < 2:
            mean_bf10_series.append(float("nan"))
            std_bf10_series.append(float("nan"))
            continue
        if mask.sum() < 2:
            mean_bf10_series.append(float("nan"))
            std_bf10_series.append(float("nan"))
            continue
        diffs = act[mask] - ind[mask]
        try:
            bf10_mean = float(compute_jzs_bf_one_sample(diffs))
        except Exception:
            bf10_mean = float("nan")
        mean_bf10_series.append(bf10_mean)
        if first_mean_idx is None and np.isfinite(bf10_mean) and bf10_mean <= bf_threshold:
            first_mean_idx = idx
        if mask.sum() < 3:
            std_bf10_series.append(float("nan"))
            continue
        sums = act[mask] + ind[mask]
        diffs_var = act[mask] - ind[mask]
        try:
            bf10_std = float(compute_jzs_bf_corr(sums, diffs_var))
        except Exception:
            bf10_std = float("nan")
        std_bf10_series.append(bf10_std)
        if first_std_idx is None and np.isfinite(bf10_std) and bf10_std <= bf_threshold:
            first_std_idx = idx
    return RMSEEquivalenceSummary(
        bf_threshold=bf_threshold,
        mean_bf10_series=mean_bf10_series,
        std_bf10_series=std_bf10_series,
        first_mean_equivalent_index=first_mean_idx,
        first_std_equivalent_index=first_std_idx,
    )


def summarize_rmse_comparison(
    participants: Dict[str, Any],
    independent_method: str,
    active_method: str,
    sample_index: int,
    *,
    equivalence_bf_threshold: float = 1.0,
) -> Tuple[RMSEPointSummary, RMSEEquivalenceSummary]:
    independent_matrix, active_matrix = _collect_rmse_pairs(participants, independent_method, active_method)
    if independent_matrix.size == 0 or active_matrix.size == 0:
        raise ValueError("No overlapping RMSE data found for requested methods.")
    if sample_index < 0 or sample_index >= independent_matrix.shape[1]:
        raise ValueError(
            f"Sample index {sample_index} is out of bounds for available RMSE length {independent_matrix.shape[1]}"
        )
    point_summary = _rmse_point_summary(
        independent_matrix[:, sample_index],
        active_matrix[:, sample_index],
        sample_index,
    )
    eq_summary = _equivalence_summary(
        independent_matrix,
        active_matrix,
        bf_threshold=equivalence_bf_threshold,
    )
    return point_summary, eq_summary

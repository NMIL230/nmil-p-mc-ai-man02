#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Sequence

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
try:
    from scipy.special import erfc as _erfc
except ImportError:  # fallback for minimal SciPy builds
    from math import erfc as _scalar_erfc
    def _erfc(x):
        return np.vectorize(_scalar_erfc)(x)

from src.utils.plotting.plotting import (
    OVERLAY_PER_PANEL_SIZE,
    OVERLAY_PANEL_W_RATIO,
    OVERLAY_PANEL_H_RATIO,
    OVERLAY_POINT_SIZE,
    OVERLAY_SUBPLOT_HSPACE,
    OVERLAY_SUBPLOT_WSPACE,
    AXIS_LABEL_FONT_SIZE,
    SUPER_TITLE_FONT_SIZE,
    SUBPLOT_TICK_LABEL_FONT_SIZE_X,
    SUBPLOT_TICK_LABEL_FONT_SIZE_Y,
    SUBPLOT_TITLE_FONT_SIZE,
    DEFAULT_FIGURE_DPI,
    _style_spines_and_ticks,
)
from src.utils.plotting.small_multiples import (
    AxisSpec,
    TitleSpec,
    apply_axis_spec,
    apply_titles_and_layout,
    create_small_multiples_layout,
    hide_extra_axes,
)
from src.utils.modeling.gp import get_axes_from_grid
from src.utils.modeling.fits import FIT_GUESS, FIT_LAPSE
from scipy.interpolate import UnivariateSpline
from src.utils.reporting.formatting import format_stat
from src.utils.plotting.constants import PSI_THETA_TEX
from src.utils.plotting.figure_naming import figure_output_stem


def _determine_grid(n: int, nrows: Optional[int] = None, ncols: Optional[int] = None) -> Tuple[int, int]:
    if n <= 0:
        return 1, 1
    if ncols is not None and ncols > 0:
        cols = int(ncols)
        rows = int(math.ceil(n / cols))
    elif nrows is not None and nrows > 0:
        rows = int(nrows)
        cols = int(math.ceil(n / rows))
    else:
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
    return max(1, rows), max(1, cols)


class ParticipantSigmoidModel:
    """Norm-CDF generative model derived from GP posteriors.

    Provides probability and sampling over (L,K) grid, with mask when undefined (L<K).
    """
    def __init__(
        self,
        entity_id: str,
        theta_values: np.ndarray,
        sigma_values: np.ndarray,
        available_mask: np.ndarray,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        guess: float = FIT_GUESS,
        lapse: float = FIT_LAPSE,
        ground_truth_contour: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        posterior_surface: Optional[np.ndarray] = None,
        posterior_x_axis: Optional[np.ndarray] = None,
        posterior_y_axis: Optional[np.ndarray] = None,
    ) -> None:
        self.entity_id = str(entity_id)
        self.theta_values = np.asarray(theta_values, dtype=float)
        self.sigma_values = np.asarray(sigma_values, dtype=float)
        self.available_mask = np.asarray(available_mask, dtype=bool)
        self.x_axis = np.asarray(x_axis, dtype=float)
        self.y_axis = np.asarray(y_axis, dtype=float)
        self.guess = float(guess)
        self.lapse = float(lapse)
        self.hyperparameters = dict(hyperparameters) if hyperparameters else {}
        self.posterior_surface = np.asarray(posterior_surface, dtype=float) if posterior_surface is not None else None
        self.posterior_x_axis = np.asarray(posterior_x_axis, dtype=float) if posterior_x_axis is not None else None
        self.posterior_y_axis = np.asarray(posterior_y_axis, dtype=float) if posterior_y_axis is not None else None
        # Store raw 50% contour (K, theta) prior to smoothing for comparison
        if ground_truth_contour is not None:
            k_raw, theta_raw = ground_truth_contour
            self.ground_truth_contour = (
                np.asarray(k_raw, dtype=float),
                np.asarray(theta_raw, dtype=float),
            )
        else:
            self.ground_truth_contour = (np.array([]), np.array([]))

    def probability(self, L: Any, K: Any) -> np.ndarray:
        L_arr = np.asarray(L, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        K_flat = np.atleast_1d(K_arr).astype(float).ravel()
        if self.y_axis.size == 0:
            return np.zeros_like(L_arr, dtype=float)
        theta_flat = np.full_like(K_flat, np.nan, dtype=float)
        sigma_flat = np.full_like(K_flat, np.nan, dtype=float)
        if np.any(self.available_mask):
            axis_vals = self.y_axis.reshape(1, -1)
            diffs = np.abs(axis_vals - K_flat.reshape(-1, 1))
            nearest_idx = np.argmin(diffs, axis=1)
            nearest_k = self.y_axis[nearest_idx]
            matches = np.isclose(nearest_k, K_flat, atol=1e-6)
            valid = matches & self.available_mask[nearest_idx]
            theta_flat[valid] = self.theta_values[nearest_idx[valid]]
            sigma_flat[valid] = self.sigma_values[nearest_idx[valid]]
        if not np.all(np.isfinite(theta_flat)):
            available_indices = np.where(self.available_mask)[0]
            if available_indices.size:
                available_ks = self.y_axis[available_indices]
                for idx, k_val in enumerate(K_flat):
                    if np.isfinite(theta_flat[idx]):
                        continue
                    nearest_idx = available_indices[np.argmin(np.abs(available_ks - k_val))]
                    theta_flat[idx] = self.theta_values[nearest_idx]
                    sigma_flat[idx] = self.sigma_values[nearest_idx]
        theta = theta_flat.reshape(K_arr.shape)
        sigma = sigma_flat.reshape(K_arr.shape)
        prob = np.zeros_like(theta, dtype=float)
        valid_mask = np.isfinite(theta) & np.isfinite(sigma)
        if np.any(valid_mask):
            sigma = np.maximum(sigma, 1e-3)
            L_broadcast = np.broadcast_to(L_arr, theta.shape)
            z = (L_broadcast - theta) / (sigma * np.sqrt(2.0))
            base_prob = 0.5 * _erfc(z)
            prob_valid = self.guess + (1.0 - self.guess - self.lapse) * base_prob
            prob_valid = np.clip(prob_valid, 0.0, 1.0)
            prob = np.where(valid_mask, prob_valid, prob)
        L_broadcast = np.broadcast_to(L_arr, theta.shape)
        K_broadcast = np.broadcast_to(K_arr, theta.shape)
        prob = np.where(L_broadcast < K_broadcast, np.nan, prob)
        return prob

    def sample(self, L: Any, K: Any, rng: Optional[np.random.Generator] = None) -> Any:
        probs = self.probability(L, K)
        if np.isnan(probs).any():
            raise ValueError("Cannot sample probabilities where L < K (undefined region).")
        rng = rng or np.random.default_rng()
        draws = rng.random(size=probs.shape)
        outcomes = (draws < probs).astype(int)
        return int(outcomes) if outcomes.shape == () else outcomes

    def evaluate_grid(self) -> np.ndarray:
        grid_xx, grid_yy = np.meshgrid(self.x_axis, self.y_axis)
        return self.probability(grid_xx, grid_yy)

    def theta_lookup(self) -> Dict[float, float]:
        available = self.available_mask & np.isfinite(self.theta_values)
        return {float(k): float(t) for k, t in zip(self.y_axis[available], self.theta_values[available])}


def posterior_slope_metrics(
    model: ParticipantSigmoidModel,
    target_k: float = 3.0,
    *,
    min_points: int = 2,
) -> Tuple[float, float, float]:
    """Estimate slope diagnostics for the 50% posterior contour.

    Returns the ΔK/ΔL slope at ``target_k`` (delta K over delta L), the mean
    ΔK/ΔL slope across all available K samples, and the threshold (θ) value at
    ``target_k``. ``NaN`` indicates that a quantity could not be
    determined for the entity.
    """

    k_vals = np.asarray(model.y_axis, dtype=float)
    theta_vals = np.asarray(model.theta_values, dtype=float)
    available = np.asarray(model.available_mask, dtype=bool)
    mask = available & np.isfinite(theta_vals) & np.isfinite(k_vals)
    if np.count_nonzero(mask) < min_points:
        return float("nan"), float("nan"), float("nan")

    k_valid = k_vals[mask]
    theta_valid = theta_vals[mask]
    if k_valid.size < min_points:
        return float("nan"), float("nan"), float("nan")

    k_unique, unique_indices = np.unique(k_valid, return_index=True)
    if k_unique.size < min_points:
        return float("nan"), float("nan"), float("nan")
    theta_unique = theta_valid[unique_indices]

    order = np.argsort(k_unique)
    k_unique = k_unique[order]
    theta_unique = theta_unique[order]

    target_slope = float("nan")
    average_slope = float("nan")
    theta_at_target = float("nan")

    try:
        dtheta_dk = np.gradient(theta_unique, k_unique)
    except Exception:
        dtheta_dk = np.full_like(theta_unique, np.nan, dtype=float)

    if np.any(np.isfinite(dtheta_dk)):
        grad_mask = np.isfinite(dtheta_dk) & (np.abs(dtheta_dk) >= 1e-6)
        if np.count_nonzero(grad_mask) >= min_points:
            slopes_all = 1.0 / dtheta_dk[grad_mask]
            if slopes_all.size:
                average_slope = float(np.nanmean(slopes_all))
        interp_mask = np.isfinite(dtheta_dk)
        if np.count_nonzero(interp_mask) >= min_points:
            k_interp = k_unique[interp_mask]
            d_interp = dtheta_dk[interp_mask]
            if target_k >= k_interp.min() and target_k <= k_interp.max():
                deriv_at_target = float(np.interp(target_k, k_interp, d_interp))
                if np.isfinite(deriv_at_target) and abs(deriv_at_target) >= 1e-6:
                    target_slope = 1.0 / deriv_at_target

    if np.any(np.isfinite(theta_unique)):
        try:
            theta_at_target = float(np.interp(target_k, k_unique, theta_unique))
        except Exception:
            pass

    if not np.isfinite(target_slope):
        distances = np.abs(k_unique - target_k)
        window_size = min(5, k_unique.size)
        window_indices = np.argsort(distances)[:window_size]
        theta_window = theta_unique[window_indices]
        k_window = k_unique[window_indices]
        if theta_window.size >= min_points:
            theta_window_ptp = float(np.ptp(theta_window)) if theta_window.size else 0.0
            if not np.isclose(theta_window_ptp, 0.0):
                X = np.vstack([theta_window, np.ones_like(theta_window)]).T
                try:
                    slope_local, _ = np.linalg.lstsq(X, k_window, rcond=None)[0]
                    target_slope = float(slope_local)
                except Exception:
                    pass

    if not np.isfinite(average_slope):
        theta_span = float(np.ptp(theta_unique)) if theta_unique.size else 0.0
        if not np.isclose(theta_span, 0.0):
            X = np.vstack([theta_unique, np.ones_like(theta_unique)]).T
            try:
                slope_global, _ = np.linalg.lstsq(X, k_unique, rcond=None)[0]
                average_slope = float(slope_global)
            except Exception:
                pass

    return target_slope, average_slope, theta_at_target


def posterior_delta_k_over_delta_l_slope(
    model: ParticipantSigmoidModel,
    target_k: float = 3.0,
    *,
    min_points: int = 2,
) -> float:
    target_slope, _avg, _theta = posterior_slope_metrics(
        model,
        target_k=target_k,
        min_points=min_points,
    )
    return target_slope


def order_models_by_posterior_slope(
    models: Dict[str, ParticipantSigmoidModel],
    target_k: float = 3.0,
    *,
    sort_mode: str = "avg",
    classic_scores: Optional[Dict[str, float]] = None,
    reverse: bool = True,
) -> Tuple[List[str], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Order entities by their ΔK/ΔL slope at the specified K value.

    Parameters
    ----------
    models
        Mapping of entity identifier to :class:`ParticipantSigmoidModel`.
    target_k
        Color-count (K) at which to evaluate the slope.
    sort_mode
        ``"avg"`` (mean ΔK/ΔL across K), ``"target"`` (ΔK/ΔL at the specified
        K), or ``"classic"`` (Classic-mode $\psi_{\theta}$). Defaults to ``"avg"``.
    reverse
        If ``True`` (default) entities with steeper positive slopes appear first.

    Returns
    -------
    order
        List of entity identifiers sorted by slope (finite slopes first).
    slopes
        Mapping of entity identifier to slope used for ordering (may be NaN).
    average_slopes
        Mapping of entity identifier to their mean ΔK/ΔL slope (may be NaN).
    theta_at_k
        Mapping of entity identifier to the θ (block threshold) at ``target_k``.
    """

    slopes: Dict[str, float] = {}
    avg_slopes: Dict[str, float] = {}
    theta_at_k: Dict[str, float] = {}
    mode_norm = sort_mode.lower()
    if mode_norm not in {"avg", "average", "target", "classic", "adaptive-3k-threshold"}:
        raise ValueError(f"Unsupported sort_mode: {sort_mode}")
    if mode_norm == "average":
        mode_norm = "avg"

    for ent, model in models.items():
        slope_target, slope_avg, theta_val = posterior_slope_metrics(
            model,
            target_k=target_k,
        )
        slopes[ent] = slope_target
        avg_slopes[ent] = slope_avg
        theta_at_k[ent] = theta_val

    classic_map: Dict[str, float] = {}
    if classic_scores:
        for pid, val in classic_scores.items():
            try:
                classic_map[str(pid)] = float(val)
            except (TypeError, ValueError):
                continue

    metrics: Dict[str, Dict[str, float]] = {
        "avg": avg_slopes,
        "target": slopes,
        "classic": classic_map,
        "theta": theta_at_k,
    }

    if mode_norm == "avg":
        metric_order = ["avg", "target", "classic", "theta"]
    elif mode_norm == "target":
        metric_order = ["target", "avg", "classic", "theta"]
    elif mode_norm == "classic":
        metric_order = ["classic", "avg", "target", "theta"]
    else:  # adaptive threshold sort
        metric_order = ["theta", "avg", "target", "classic"]

    def metric_value(metric_name: str, ent: str) -> float:
        mapping = metrics.get(metric_name) or {}
        try:
            return float(mapping.get(ent, float("nan")))
        except (TypeError, ValueError):
            return float("nan")

    def value_to_key(val: float) -> Tuple[int, float]:
        if np.isfinite(val):
            sort_val = -val if reverse else val
            return (0, sort_val)
        return (1, float("inf"))

    def sort_key(ent: str) -> Tuple[Any, ...]:
        parts: List[Any] = []
        for name in metric_order:
            parts.extend(value_to_key(metric_value(name, ent)))
        parts.append(ent)
        return tuple(parts)

    ordered = sorted(models.keys(), key=sort_key)
    return ordered, slopes, avg_slopes, theta_at_k


def plot_adaptive_scatter_with_posterior50(
    points_by_entity: Dict[str, List[Tuple[float, float, int]]],
    gp_by_entity: Dict[str, List[dict]],
    output_dir: Path,
    title: str = "Adaptive Outcomes and 50% Posterior",
    filename_stem: str = f"{figure_output_stem('Fig03')}_adaptive_scatter_50",
    per_panel_size: float = OVERLAY_PER_PANEL_SIZE,
    entity_order: Sequence[str] | None = None,
    sort_metric: Optional[Dict[str, float]] = None,
    sort_metric_name: Optional[str] = None,
    show_sort_metric: bool = True,
    classic_thresholds: Optional[Dict[str, float]] = None,
    show_classic_threshold: bool = True,
    nrows_override: Optional[int] = None,
    ncols_override: Optional[int] = None,
) -> Path:
    combined = set(points_by_entity.keys()) | set(gp_by_entity.keys())
    entities = sorted(combined)
    if entity_order:
        preferred = [ent for ent in entity_order if ent in combined]
        remaining = [ent for ent in entities if ent not in preferred]
        entities = preferred + remaining
    n = len(entities)
    if n == 0:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{filename_stem}.png"

    layout = create_small_multiples_layout(
        n,
        nrows_override=nrows_override,
        ncols_override=ncols_override,
        include_legend_slot=True,
        per_panel_size=per_panel_size,
        panel_w_ratio=OVERLAY_PANEL_W_RATIO,
        panel_h_ratio=OVERLAY_PANEL_H_RATIO,
    )
    fig = layout.fig
    axes_flat = layout.panel_axes

    axis_spec = AxisSpec(
        x_min=1.0,
        x_max=17.0,
        y_min=1.0,
        y_max=8.0,
        x_ticks=range(1, 18, 2),
        y_ticks=range(1, 9),
        x_buffer_frac=0.05,
        y_buffer_frac=0.05,
    )

    fail_present = False
    pass_present = False
    threshold_present = False

    for idx, ent in enumerate(entities):
        row, col = layout.positions[idx]
        ax = axes_flat[idx]
        pts = points_by_entity.get(ent, []) or []
        if pts:
            arr = np.asarray(pts, dtype=float)
            x_pts = arr[:, 0]
            y_pts = arr[:, 1]
            y_lab = arr[:, 2]
        else:
            x_pts = np.array([])
            y_pts = np.array([])
            y_lab = np.array([])

        if y_lab.size:
            sel_pass = y_lab >= 0.5
            sel_fail = ~sel_pass
            if np.any(sel_fail):
                ax.scatter(x_pts[sel_fail], y_pts[sel_fail], s=OVERLAY_POINT_SIZE, facecolors='none', edgecolors='crimson', linewidths=1.2, alpha=1.0, marker='D', zorder=3)
                fail_present = True
            if np.any(sel_pass):
                ax.scatter(x_pts[sel_pass], y_pts[sel_pass], s=OVERLAY_POINT_SIZE, color='royalblue', alpha=0.9, marker='+', linewidths=1.2, zorder=3)
                pass_present = True

        gp_list = gp_by_entity.get(ent, []) or []
        grid_xx_ref: Optional[np.ndarray] = None
        grid_yy_ref: Optional[np.ndarray] = None
        z_list = []
        for gp in gp_list:
            try:
                grid_xx = np.asarray(gp["grid_xx"], dtype=float)
                grid_yy = np.asarray(gp["grid_yy"], dtype=float)
                zz = np.asarray(gp["zz_posterior"], dtype=float)
                if zz.ndim == 1:
                    zz = zz.reshape(grid_xx.shape)
                if grid_xx_ref is None or grid_yy_ref is None:
                    grid_xx_ref = grid_xx
                    grid_yy_ref = grid_yy
                elif (grid_xx.shape != grid_xx_ref.shape or grid_yy.shape != grid_yy_ref.shape):
                    continue
                z_list.append(zz)
            except Exception:
                continue
        if z_list and grid_xx_ref is not None and grid_yy_ref is not None:
            mean_zz = np.nanmean(np.stack(z_list, axis=0), axis=0)
            try:
                zmin = float(np.nanmin(mean_zz))
                zmax = float(np.nanmax(mean_zz))
                if (zmin <= 0.5 <= zmax) and np.isfinite(zmin) and np.isfinite(zmax):
                    ax.contour(grid_xx_ref, grid_yy_ref, mean_zz, levels=[0.5], colors='black', linewidths=1.2)
            except Exception:
                pass

        subplot_title = str(ent)
        if show_sort_metric and sort_metric is not None and ent in sort_metric:
            metric_val = float(sort_metric.get(ent, float("nan")))
            label = sort_metric_name or "metric"
            if np.isfinite(metric_val):
                metric_fmt = format_stat(metric_val, mode="decimal")
                subplot_title = f"{ent}\n{label}={metric_fmt}"
            else:
                subplot_title = f"{ent}\n{label}=nan"
        ax.set_title(subplot_title, fontsize=SUBPLOT_TITLE_FONT_SIZE)

        if show_classic_threshold and classic_thresholds is not None:
            c_val = float(classic_thresholds.get(ent, float("nan")))
            if np.isfinite(c_val):
                ax.scatter([c_val], [3.0], marker='o', color='purple', s=OVERLAY_POINT_SIZE * 1.8, alpha=0.8, edgecolors='none', zorder=4)
                threshold_present = True

        apply_axis_spec(ax, axis_spec)
        _style_spines_and_ticks(ax)
        ax.tick_params(axis='x', labelsize=SUBPLOT_TICK_LABEL_FONT_SIZE_X)
        ax.tick_params(axis='y', labelsize=SUBPLOT_TICK_LABEL_FONT_SIZE_Y)
        if row != layout.bottom_row_for_col.get(col, row):
            ax.tick_params(labelbottom=False)
        if col != 0:
            ax.tick_params(labelleft=False)

    legend_ax = layout.legend_ax
    if legend_ax is not None:
        legend_ax.axis('off')
        legend_handles: List[Any] = []
        legend_labels: List[str] = []
        if fail_present:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color='crimson',
                    marker='D',
                    linestyle='None',
                    markersize=6,
                    markerfacecolor='none',
                    markeredgewidth=1.2,
                )
            )
            legend_labels.append('Fail')
        if pass_present:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color='royalblue',
                    marker='+',
                    linestyle='None',
                    markersize=7,
                    markeredgewidth=1.2,
                )
            )
            legend_labels.append('Pass')
        if show_classic_threshold and classic_thresholds is not None and threshold_present:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color='purple',
                    marker='o',
                    linestyle='None',
                    markersize=7,
                    markerfacecolor='purple',
                    markeredgewidth=0,
                    alpha=0.8,
                )
            )
            legend_labels.append(f"Classic {PSI_THETA_TEX} (K=3)")
        if legend_handles:
            legend_ax.legend(legend_handles, legend_labels, loc='center', frameon=False)
        else:
            legend_ax.axis('off')

    hide_extra_axes(layout)

    apply_titles_and_layout(
        layout,
        TitleSpec(
            title="Individual Session Thresholds",
            title_fontsize=SUPER_TITLE_FONT_SIZE * 2.2,
            title_y=0.955,
            xlabel="Pattern Size (Blocks)",
            xlabel_fontsize=AXIS_LABEL_FONT_SIZE * 1.8,
            xlabel_y=0.075,
            ylabel="Number of Unique Colors",
            ylabel_fontsize=AXIS_LABEL_FONT_SIZE * 1.8,
            ylabel_x=0.055,
            tight_layout_rect=(0.06, 0.08, 0.99, 0.96),
            subplot_wspace=OVERLAY_SUBPLOT_WSPACE,
            subplot_hspace=OVERLAY_SUBPLOT_HSPACE,
        ),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{filename_stem}.{ext}", dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return Path(output_dir) / f"{filename_stem}.png"
def build_norm_cdf_generators(
    gp_by_entity: Dict[str, List[dict]],
    classic_fits: Dict[str, List[Dict[str, float]]],
    *,
    spread_mode: str = "fixed",
    fixed_sigma: float = 2.0,
    threshold: float = 0.5,
    min_sigma: float = 0.2,
    smooth_k_spline: bool = True,
    spline_s: float = 0.5,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, ParticipantSigmoidModel]:
    spreads: List[float] = []
    if spread_mode.lower() == "cm_average":
        for fits in classic_fits.values():
            for fit in fits:
                try:
                    val = float(fit.get("spread", float("nan")))
                    if np.isfinite(val) and val > 0:
                        spreads.append(val)
                except Exception:
                    continue
    sigma_default = float(fixed_sigma)
    classic_spread_mean: Optional[float] = None
    if spreads:
        classic_spread_mean = float(np.mean(spreads))
        sigma_default = float(classic_spread_mean)
    sigma_default = max(min_sigma, sigma_default)
    models: Dict[str, ParticipantSigmoidModel] = {}
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({
            "spread_mode": spread_mode,
            "fixed_sigma": float(fixed_sigma),
            "min_sigma": float(min_sigma),
            "classic_spread_mean": classic_spread_mean,
            "sigma_default": sigma_default,
            "entities": {},
            "smooth_k_spline": bool(smooth_k_spline),
            "spline_s": float(spline_s),
        })
    for ent, gp_list in gp_by_entity.items():
        grid_xx_ref: Optional[np.ndarray] = None
        grid_yy_ref: Optional[np.ndarray] = None
        surfaces: List[np.ndarray] = []
        hyperparameters: Optional[Dict[str, Any]] = None
        for gp in gp_list:
            try:
                grid_xx = np.asarray(gp["grid_xx"], dtype=float)
                grid_yy = np.asarray(gp["grid_yy"], dtype=float)
                zz = np.asarray(gp["zz_posterior"], dtype=float)
                if zz.ndim == 1:
                    zz = zz.reshape(grid_xx.shape)
                if grid_xx_ref is None or grid_yy_ref is None:
                    grid_xx_ref = grid_xx
                    grid_yy_ref = grid_yy
                elif (grid_xx.shape != grid_xx_ref.shape or grid_yy.shape != grid_yy_ref.shape):
                    continue
                surfaces.append(zz)
                if hyperparameters is None:
                    hp = gp.get("model_hyperparameters")
                    if isinstance(hp, dict):
                        hyperparameters = dict(hp)
            except Exception:
                continue
        if not surfaces or grid_xx_ref is None or grid_yy_ref is None:
            continue
        stacked = np.stack(surfaces, axis=0)
        if not np.any(np.isfinite(stacked)):
            continue
        mean_surface = np.nanmean(stacked, axis=0)
        x_axis, y_axis = get_axes_from_grid(grid_xx_ref, grid_yy_ref)
        valid_cols = np.isfinite(x_axis) & np.any(np.isfinite(mean_surface), axis=0)
        valid_rows = np.isfinite(y_axis) & np.any(np.isfinite(mean_surface), axis=1)
        if np.count_nonzero(valid_cols) < 2 or np.count_nonzero(valid_rows) < 2:
            continue
        surface_clean = mean_surface[np.ix_(valid_rows, valid_cols)]
        x_axis_clean = x_axis[valid_cols]
        y_axis_clean = y_axis[valid_rows]
        # Derive 50% crossing per color
        theta_vals = np.full_like(y_axis, np.nan, dtype=float)
        available = np.zeros_like(y_axis, dtype=bool)
        valid_row_indices = np.where(valid_rows)[0]
        for trim_idx, row_idx in enumerate(valid_row_indices):
            row = surface_clean[trim_idx, :]
            mask = np.isfinite(row)
            if not np.any(mask):
                continue
            x = x_axis_clean[mask]
            y = row[mask] - float(threshold)
            if x.size == 0:
                continue
            if np.all(y > 0) or np.all(y < 0):
                continue
            zero_mask = np.isclose(y, 0.0)
            if np.any(zero_mask):
                theta = float(x[zero_mask][0])
            else:
                theta = None
                for idx in range(y.size - 1):
                    y1 = y[idx]
                    y2 = y[idx + 1]
                    if y1 * y2 < 0:
                        x1 = x[idx]
                        x2 = x[idx + 1]
                        ratio = y1 / (y1 - y2)
                        theta = float(x1 + ratio * (x2 - x1))
                        break
            if theta is None:
                continue
            theta_vals[row_idx] = float(theta)
            available[row_idx] = True
        # Capture raw contour (before smoothing)
        k_raw = y_axis[available]
        theta_raw = theta_vals[available]
        sigma_values = np.full_like(theta_vals, sigma_default, dtype=float)
        # Optional smoothing: fit a spline to theta(K) across available K values
        if smooth_k_spline:
            try:
                k_vals = y_axis[available]
                t_vals = theta_vals[available]
                if k_vals.size >= 3:
                    # Spline order k up to 3 but limited by number of points
                    order = 3 if k_vals.size >= 4 else (2 if k_vals.size >= 3 else 1)
                    spline = UnivariateSpline(k_vals.astype(float), t_vals.astype(float), s=float(spline_s), k=order)
                    theta_smooth = spline(y_axis.astype(float))
                    # Clip to observed x-axis bounds
                    lo, hi = float(np.nanmin(x_axis)), float(np.nanmax(x_axis))
                    theta_vals = np.clip(theta_smooth, lo, hi)
                    available = np.isfinite(theta_vals)
            except Exception:
                pass

        models[ent] = ParticipantSigmoidModel(
            entity_id=str(ent),
            theta_values=theta_vals,
            sigma_values=sigma_values,
            available_mask=available,
            x_axis=x_axis.astype(float),
            y_axis=y_axis.astype(float),
            guess=FIT_GUESS,
            lapse=FIT_LAPSE,
            ground_truth_contour=(k_raw, theta_raw),
            hyperparameters=hyperparameters,
            posterior_surface=surface_clean,
            posterior_x_axis=x_axis_clean,
            posterior_y_axis=y_axis_clean,
        )
    return models


def plot_adaptive_probability_surface(
    models: Dict[str, ParticipantSigmoidModel],
    output_dir: Path,
    filename_stem: str = "adaptive_prob_surface",
    diagnostics: Optional[Dict[str, Any]] = None,
    entity_order: Sequence[str] | None = None,
    sort_metric: Optional[Dict[str, float]] = None,
    sort_metric_name: Optional[str] = None,
    show_sort_metric: bool = True,
    nrows_override: Optional[int] = None,
    ncols_override: Optional[int] = None,
    use_gp_surface: bool = True,
    points_by_entity: Optional[Dict[str, List[Tuple[float, float, int]]]] = None,
    classic_thresholds: Optional[Dict[str, float]] = None,
    show_classic_threshold: bool = False,
) -> Path:
    entities = list(models.keys())
    if entity_order:
        preferred = [ent for ent in entity_order if ent in models]
        remaining = [ent for ent in entities if ent not in preferred]
        entities = preferred + remaining
    n = len(entities)
    if diagnostics is not None:
        diagnostics["failure_count"] = 0
        diagnostics["failures"] = {}
    if n == 0:
        return Path(output_dir) / f"{filename_stem}.png"

    layout = create_small_multiples_layout(
        n,
        nrows_override=nrows_override,
        ncols_override=ncols_override,
        include_legend_slot=True,
        per_panel_size=OVERLAY_PER_PANEL_SIZE,
        panel_w_ratio=OVERLAY_PANEL_W_RATIO,
        panel_h_ratio=OVERLAY_PANEL_H_RATIO,
    )
    fig = layout.fig
    axes_flat = layout.panel_axes

    axis_spec = AxisSpec(
        x_min=1.0,
        x_max=17.0,
        y_min=1.0,
        y_max=8.0,
        x_ticks=range(1, 18, 2),
        y_ticks=range(1, 9),
        x_buffer_frac=0.05,
        y_buffer_frac=0.05,
    )

    normcdf_surface_used = False
    posterior_surface_used = False
    fail_present = False
    pass_present = False
    threshold_present = False

    for idx, ent in enumerate(entities):
        row, col = layout.positions[idx]
        ax = axes_flat[idx]
        model = models[ent]
        plotted_surface = False
        if use_gp_surface:
            posterior = getattr(model, "posterior_surface", None)
            posterior_x = getattr(model, "posterior_x_axis", None)
            posterior_y = getattr(model, "posterior_y_axis", None)
            if posterior is not None and posterior_x is not None and posterior_y is not None:
                try:
                    post_arr = np.asarray(posterior, dtype=float)
                    x_post = np.asarray(posterior_x, dtype=float)
                    y_post = np.asarray(posterior_y, dtype=float)
                    if post_arr.ndim == 2 and x_post.ndim == 1 and y_post.ndim == 1 and post_arr.shape == (y_post.size, x_post.size):
                        finite_x = x_post[np.isfinite(x_post)]
                        finite_y = y_post[np.isfinite(y_post)]
                        if finite_x.size and finite_y.size and post_arr.size:
                            x_min, x_max = float(np.nanmin(finite_x)), float(np.nanmax(finite_x))
                            y_min, y_max = float(np.nanmin(finite_y)), float(np.nanmax(finite_y))
                            ax.imshow(post_arr, origin='lower', aspect='auto', extent=[x_min, x_max, y_min, y_max], cmap='Greys_r', vmin=0.0, vmax=1.0)
                            try:
                                xx, yy = np.meshgrid(x_post, y_post)
                                ax.contour(xx, yy, post_arr, levels=[0.5], colors='black', linewidths=1.2)
                            except Exception:
                                pass
                            posterior_surface_used = True
                            plotted_surface = True
                except Exception:
                    pass
        if not plotted_surface:
            try:
                grid = model.evaluate_grid()
                x = model.x_axis
                y = model.y_axis
                extent = [x.min(), x.max(), y.min(), y.max()]
                ax.imshow(grid, origin='lower', aspect='auto', extent=extent, cmap='Greys_r', vmin=0.0, vmax=1.0)
                ax.contour(x.reshape(1, -1), y.reshape(-1, 1), grid, levels=[0.5], colors='black', linewidths=1.2)
                normcdf_surface_used = True
                plotted_surface = True
            except Exception:
                if diagnostics is not None:
                    diagnostics["failure_count"] += 1
                    diagnostics["failures"][ent] = "surface eval error"
                ax.text(0.5, 0.5, 'No surface', transform=ax.transAxes, ha='center', va='center')

        pts: List[Tuple[float, float, int]] = []
        if points_by_entity is not None:
            pts = points_by_entity.get(ent, []) or []
        if pts:
            try:
                arr = np.asarray(pts, dtype=float)
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    x_pts = arr[:, 0]
                    y_pts = arr[:, 1]
                    outcomes = arr[:, 2]
                    sel_pass = outcomes >= 0.5
                    sel_fail = ~sel_pass
                    if np.any(sel_fail):
                        ax.scatter(
                            x_pts[sel_fail],
                            y_pts[sel_fail],
                            s=OVERLAY_POINT_SIZE,
                            facecolors='none',
                            edgecolors='crimson',
                            linewidths=1.2,
                            alpha=1.0,
                            marker='D',
                            zorder=4,
                        )
                        fail_present = True
                    if np.any(sel_pass):
                        ax.scatter(
                            x_pts[sel_pass],
                            y_pts[sel_pass],
                            s=OVERLAY_POINT_SIZE,
                            color='royalblue',
                            alpha=0.9,
                            marker='+',
                            linewidths=1.2,
                            zorder=4,
                        )
                        pass_present = True
            except Exception:
                pass

        subplot_title = str(ent)
        if show_sort_metric and sort_metric is not None and ent in sort_metric:
            metric_val = float(sort_metric.get(ent, float("nan")))
            label = sort_metric_name or "metric"
            if np.isfinite(metric_val):
                metric_fmt = format_stat(metric_val, mode="decimal")
                subplot_title = f"{ent}\n{label}={metric_fmt}"
            else:
                subplot_title = f"{ent}\n{label}=nan"
        ax.set_title(subplot_title, fontsize=SUBPLOT_TITLE_FONT_SIZE)

        if show_classic_threshold and classic_thresholds is not None:
            c_val = float(classic_thresholds.get(ent, float("nan")))
            if np.isfinite(c_val):
                ax.scatter(
                    [c_val],
                    [3.0],
                    marker='o',
                    color='purple',
                    s=OVERLAY_POINT_SIZE * 1.8,
                    alpha=0.8,
                    edgecolors='none',
                    zorder=5,
                )
                threshold_present = True

        apply_axis_spec(ax, axis_spec)
        _style_spines_and_ticks(ax)
        ax.tick_params(axis='x', labelsize=SUBPLOT_TICK_LABEL_FONT_SIZE_X)
        ax.tick_params(axis='y', labelsize=SUBPLOT_TICK_LABEL_FONT_SIZE_Y)
        if row != layout.bottom_row_for_col.get(col, row):
            ax.tick_params(labelbottom=False)
        if col != 0:
            ax.tick_params(labelleft=False)

    legend_ax = layout.legend_ax
    if legend_ax is not None:
        legend_ax.axis('off')
        legend_handles = []
        legend_labels = []
        if fail_present:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color='crimson',
                    marker='D',
                    linestyle='None',
                    markersize=6,
                    markerfacecolor='none',
                    markeredgewidth=1.2,
                )
            )
            legend_labels.append('Fail')
        if pass_present:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color='royalblue',
                    marker='+',
                    linestyle='None',
                    markersize=7,
                    markeredgewidth=1.2,
                )
            )
            legend_labels.append('Pass')
        if show_classic_threshold and classic_thresholds is not None and threshold_present:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color='purple',
                    marker='o',
                    linestyle='None',
                    markersize=7,
                    markerfacecolor='purple',
                    markeredgewidth=0,
                    alpha=0.8,
                )
            )
            legend_labels.append(f"Classic {PSI_THETA_TEX} (K=3)")
        if posterior_surface_used:
            legend_handles.append(Line2D([0], [0], color='black', lw=1.2, label='posterior 0.5 contour'))
            legend_labels.append('posterior 0.5 contour')
        if normcdf_surface_used:
            label = 'norm-CDF 0.5 contour' if posterior_surface_used else '0.5 contour'
            legend_handles.append(Line2D([0], [0], color='black', lw=1.2, label=label))
            legend_labels.append(label)
        if legend_handles:
            legend_ax.legend(legend_handles, legend_labels, loc='center', frameon=False)
        else:
            legend_ax.axis('off')

    hide_extra_axes(layout)

    apply_titles_and_layout(
        layout,
        TitleSpec(
            title="Individual Session Thresholds",
            title_fontsize=SUPER_TITLE_FONT_SIZE * 2.2,
            title_y=0.955,
            xlabel="Pattern Size (Blocks)",
            xlabel_fontsize=AXIS_LABEL_FONT_SIZE * 1.8,
            xlabel_y=0.075,
            ylabel="Number of Unique Colors",
            ylabel_fontsize=AXIS_LABEL_FONT_SIZE * 1.8,
            ylabel_x=0.055,
            tight_layout_rect=(0.06, 0.08, 0.99, 0.96),
            subplot_wspace=OVERLAY_SUBPLOT_WSPACE,
            subplot_hspace=OVERLAY_SUBPLOT_HSPACE,
        ),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{filename_stem}.{ext}", dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return Path(output_dir) / f"{filename_stem}.png"

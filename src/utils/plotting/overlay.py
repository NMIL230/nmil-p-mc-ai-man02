#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Optional, TYPE_CHECKING, Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.utils.modeling.fits import logistic, FIT_GUESS, FIT_LAPSE
from matplotlib.lines import Line2D
from src.utils.plotting.plotting import (
    TYPE_X_COLOR,
    TYPE_Y_COLOR,
    OVERLAY_POINT_SIZE,
    OVERLAY_POINT_ALPHA,
    OVERLAY_LINE_WIDTH,
    OVERLAY_PER_PANEL_SIZE,
    OVERLAY_PANEL_W_RATIO,
    OVERLAY_PANEL_H_RATIO,
    OVERLAY_SUBPLOT_HSPACE,
    OVERLAY_SUBPLOT_WSPACE,
    CH_CLASSIC_LABEL,
    CH_ADAPTIVE_LABEL,
    AXIS_LABEL_FONT_SIZE,
    SUBPLOT_TICK_LABEL_FONT_SIZE_X,
    SUBPLOT_TICK_LABEL_FONT_SIZE_Y,
    SUBPLOT_TITLE_FONT_SIZE,
    SUPER_TITLE_FONT_SIZE,
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
from src.utils.reporting.formatting import format_stat


if TYPE_CHECKING:  # pragma: no cover - for typing only
    from src.utils.modeling.adaptive import ParticipantSigmoidModel


# Default posterior slice for adaptive overlays (Number of Unique Colors = 3)
POSTERIOR_SLICE_K = 3.0
CLASSIC_INTERVAL_Y = 0.52
ADAPTIVE_INTERVAL_Y = 0.48

def _adjusted_probability(entry: Dict[str, float]) -> float:
    """Convert successes/trials to a guess/lapse-adjusted probability."""

    successes = float(entry.get("successes", 0.0))
    trials = float(entry.get("trials", 0.0))
    base = successes / max(1.0, trials)
    prob = FIT_GUESS + (1.0 - FIT_GUESS - FIT_LAPSE) * base
    return float(np.clip(prob, 0.0, 1.0))


def _posterior_slice_curve(
    model: "ParticipantSigmoidModel",
    target_k: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract a 1D posterior slice at the specified K value."""

    surface = getattr(model, "posterior_surface", None)
    x_axis = getattr(model, "posterior_x_axis", None)
    y_axis = getattr(model, "posterior_y_axis", None)
    if surface is None or x_axis is None or y_axis is None:
        return None, None
    try:
        surface_arr = np.asarray(surface, dtype=float)
        x_arr = np.asarray(x_axis, dtype=float)
        y_arr = np.asarray(y_axis, dtype=float)
    except Exception:
        return None, None
    if surface_arr.ndim != 2 or x_arr.ndim != 1 or y_arr.ndim != 1:
        return None, None
    # Ensure orientation is (len(y), len(x))
    if surface_arr.shape != (y_arr.size, x_arr.size):
        if surface_arr.shape == (x_arr.size, y_arr.size):
            surface_arr = surface_arr.T
        else:
            return None, None
    if not np.any(np.isfinite(y_arr)) or not np.any(np.isfinite(surface_arr)):
        return None, None
    idx = int(np.nanargmin(np.abs(y_arr - target_k)))
    slice_vals = surface_arr[idx, :]
    if not np.any(np.isfinite(slice_vals)):
        return None, None
    return x_arr, slice_vals


def _logistic_interval_x(
    fit: Dict[str, float],
    lower: float,
    upper: float,
) -> Optional[Tuple[float, float]]:
    x0 = fit.get("psi_theta")
    spread = fit.get("spread")
    if x0 is None or spread in (None, 0):
        return None
    x0 = float(x0)
    spread = float(spread)
    g = float(FIT_GUESS)
    l = float(FIT_LAPSE)
    denom = 1.0 - g - l
    if denom <= 0:
        return None
    k = 2.0 * np.log(3.0) / spread

    def _invert(prob: float) -> Optional[float]:
        s = (prob - g) / denom
        if s <= 0.0 or s >= 1.0:
            return None
        return x0 + (1.0 / k) * np.log((1.0 - s) / s)

    x_lower = _invert(lower)
    x_upper = _invert(upper)
    if x_lower is None or x_upper is None:
        return None
    lo, hi = sorted((float(x_lower), float(x_upper)))
    return lo, hi


def _posterior_interval_x(
    x_vals: np.ndarray,
    probs: np.ndarray,
    lower: float,
    upper: float,
) -> Optional[Tuple[float, float]]:
    def _crossing(target: float, from_right: bool = False) -> Optional[float]:
        valid = np.isfinite(x_vals) & np.isfinite(probs)
        if not np.any(valid):
            return None
        x = x_vals[valid]
        y = probs[valid] - target
        if x.size < 2:
            idx = np.where(np.isclose(y, 0.0, atol=1e-6))[0]
            return float(x[idx[0]]) if idx.size else None
        sign_changes = np.where(np.diff(np.sign(y)) != 0)[0]
        if sign_changes.size == 0:
            idx = np.where(np.isclose(y, 0.0, atol=1e-6))[0]
            if idx.size:
                return float(x[idx[0]]) if not from_right else float(x[idx[-1]])
            return None
        idx0 = sign_changes[-1] if from_right else sign_changes[0]
        x0, x1 = x[idx0], x[idx0 + 1]
        y0, y1 = y[idx0], y[idx0 + 1]
        if y1 == y0:
            return float(x0)
        return float(x0 - y0 * (x1 - x0) / (y1 - y0))

    x_lower = _crossing(lower, from_right=False)
    x_upper = _crossing(upper, from_right=True)
    if x_lower is None or x_upper is None:
        return None
    if x_lower > x_upper:
        x_lower, x_upper = x_upper, x_lower
    return x_lower, x_upper


def plot_sigmoid_grid_overlay(
    fits_x_by_entity: Dict[str, List[Dict[str, float]]],
    fits_y_by_entity: Dict[str, List[Dict[str, float]]],
    *,
    output_dir: Path,
    filename_stem: str = "sigmoids_overlay",
    per_panel_size: float = OVERLAY_PER_PANEL_SIZE,
    entity_order: Sequence[str] | None = None,
    sort_metric: Optional[Dict[str, float]] = None,
    sort_metric_name: Optional[str] = None,
    show_sort_metric: bool = False,
    nrows_override: Optional[int] = None,
    ncols_override: Optional[int] = None,
    posterior_models: Optional[Dict[str, "ParticipantSigmoidModel"]] = None,
    posterior_slice_k: float = POSTERIOR_SLICE_K,
) -> Path:
    combined = set(fits_x_by_entity.keys()) | set(fits_y_by_entity.keys())
    entities = sorted(combined)
    if entity_order:
        preferred = [ent for ent in entity_order if ent in combined]
        remaining = [ent for ent in entities if ent not in preferred]
        entities = preferred + remaining
    n = len(entities)
    if n == 0:
        return Path(output_dir) / f"{filename_stem}.png"
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
        x_min=3.0,
        x_max=17.0,
        y_min=0.0,
        y_max=1.0,
        x_ticks=range(3, 18, 2),
        y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        x_buffer_frac=0.05,
        y_buffer_frac=0.0,
    )
    x_line = np.linspace(axis_spec.x_min, axis_spec.x_max, 200)

    prob_interval = (0.25, 0.75)

    for idx, ent in enumerate(entities):
        row, col = layout.positions[idx]
        ax = axes_flat[idx]

        # Classic (blue) fit and scatter
        classic_fits = fits_x_by_entity.get(ent, []) or []
        for fit in classic_fits:
            dm = fit.get("data")
            if not dm:
                continue
            xs = np.array(sorted(dm.keys()), dtype=float)
            if xs.size == 0:
                continue
            ys = np.array([_adjusted_probability(dm[x]) for x in xs], dtype=float)
            ax.scatter(xs, ys, s=OVERLAY_POINT_SIZE, color=TYPE_X_COLOR, alpha=OVERLAY_POINT_ALPHA, zorder=3)
            x0 = fit.get("psi_theta")
            sp = fit.get("spread", 2.0)
            if x0 is not None and sp is not None:
                x0 = float(x0)
                sp = float(sp)
                curve = logistic(x_line, x0, sp) / 100.0
                ax.plot(x_line, curve, color=TYPE_X_COLOR, linewidth=OVERLAY_LINE_WIDTH, zorder=2)
                interval = _logistic_interval_x(fit, *prob_interval)
                if interval is not None:
                    lo, hi = interval
                    lo = max(axis_spec.x_min, lo)
                    hi = min(axis_spec.x_max, hi)
                    if lo < hi:
                        ax.hlines(y=CLASSIC_INTERVAL_Y, xmin=lo, xmax=hi, colors=TYPE_X_COLOR, linewidth=2.0, zorder=1)
            break

        adaptive_fit: Optional[Dict[str, float]] = None
        for fit in fits_y_by_entity.get(ent, []) or []:
            if fit.get("psi_theta") is None:
                continue
            adaptive_fit = fit
            break

        adaptive_curve_plotted = False
        posterior_interval = None
        if posterior_models and ent in posterior_models:
            x_curve, posterior_curve = _posterior_slice_curve(posterior_models[ent], posterior_slice_k)
            if x_curve is not None and posterior_curve is not None:
                ax.plot(x_curve, posterior_curve, color=TYPE_Y_COLOR, linewidth=OVERLAY_LINE_WIDTH, zorder=2)
                interval = _posterior_interval_x(x_curve, posterior_curve, *prob_interval)
                if interval is not None:
                    posterior_interval = interval
                adaptive_curve_plotted = True

        if not adaptive_curve_plotted and adaptive_fit is not None:
            x0 = adaptive_fit.get("psi_theta")
            sp = adaptive_fit.get("spread", 2.0)
            if x0 is not None and sp is not None:
                curve = logistic(x_line, float(x0), float(sp)) / 100.0
                ax.plot(x_line, curve, color=TYPE_Y_COLOR, linewidth=OVERLAY_LINE_WIDTH, zorder=2)
                interval = _logistic_interval_x(adaptive_fit, *prob_interval)
                if interval is not None:
                    posterior_interval = interval

        if posterior_interval is not None:
            lo, hi = posterior_interval
            lo = max(axis_spec.x_min, lo)
            hi = min(axis_spec.x_max, hi)
            if lo < hi:
                ax.hlines(y=ADAPTIVE_INTERVAL_Y, xmin=lo, xmax=hi, colors=TYPE_Y_COLOR, linewidth=2.0, zorder=1)

        title = str(ent)
        if show_sort_metric and sort_metric is not None and ent in sort_metric:
            val = float(sort_metric.get(ent, float("nan")))
            metric_label = sort_metric_name or "metric"
            if np.isfinite(val):
                metric_fmt = format_stat(val, mode="decimal")
                title = f"{ent}\n{metric_label}={metric_fmt}"
            else:
                title = f"{ent}\n{metric_label}=nan"
        ax.set_title(title, fontsize=SUBPLOT_TITLE_FONT_SIZE)

        apply_axis_spec(ax, axis_spec)
        ax.set_yticklabels(["0", ".25", ".50", ".75", "1"])
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
        handles = [
            Line2D([0], [0], color=TYPE_X_COLOR, lw=2, label=CH_CLASSIC_LABEL),
            Line2D([0], [0], color=TYPE_Y_COLOR, lw=2, label=CH_ADAPTIVE_LABEL),
        ]
        legend_ax.legend(handles=handles, loc='center', frameon=False)

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
            ylabel="Probability Correct",
            ylabel_fontsize=AXIS_LABEL_FONT_SIZE * 1.8,
            ylabel_x=0.055,
            tight_layout_rect=(0.06, 0.08, 0.99, 0.96),
            subplot_wspace=OVERLAY_SUBPLOT_WSPACE,
            subplot_hspace=OVERLAY_SUBPLOT_HSPACE,
        ),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{filename_stem}.png"
    pdf_path = output_dir / f"{filename_stem}.pdf"
    fig.savefig(png_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return png_path

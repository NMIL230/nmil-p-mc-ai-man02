#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.utils.plotting.plotting import (
    OVERLAY_PER_PANEL_SIZE,
    OVERLAY_PANEL_W_RATIO,
    OVERLAY_PANEL_H_RATIO,
)


@dataclass(frozen=True)
class AxisSpec:
    """Axis configuration shared across small-multiple figures."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    x_ticks: Sequence[float]
    y_ticks: Sequence[float]
    x_buffer_frac: float = 0.05
    y_buffer_frac: float = 0.05


@dataclass
class TitleSpec:
    """Figure-level title and label configuration."""

    title: Optional[str] = None
    title_fontsize: Optional[float] = None
    title_y: float = 0.955
    xlabel: Optional[str] = None
    xlabel_fontsize: Optional[float] = None
    xlabel_y: float = 0.075
    ylabel: Optional[str] = None
    ylabel_fontsize: Optional[float] = None
    ylabel_x: float = 0.055
    tight_layout_rect: Tuple[float, float, float, float] = (0.06, 0.08, 0.99, 0.96)
    tight_layout_pad: float = 1.08
    tight_layout_w_pad: Optional[float] = None
    tight_layout_h_pad: Optional[float] = None
    subplot_wspace: Optional[float] = None
    subplot_hspace: Optional[float] = None


@dataclass
class SmallMultiplesLayout:
    """Container describing the shared subplot grid layout."""

    fig: plt.Figure
    panel_axes: np.ndarray
    legend_ax: Optional[plt.Axes]
    extra_axes: Sequence[plt.Axes]
    positions: List[Tuple[int, int]]
    bottom_row_for_col: Dict[int, int]
    nrows: int
    ncols: int


def determine_grid(n: int, nrows_override: Optional[int] = None, ncols_override: Optional[int] = None) -> Tuple[int, int]:
    """Compute the subplot grid dimensions, honoring any overrides."""

    if n <= 0:
        return 1, 1
    if ncols_override is not None and ncols_override > 0:
        cols = int(ncols_override)
        rows = int(np.ceil(n / cols))
    elif nrows_override is not None and nrows_override > 0:
        rows = int(nrows_override)
        cols = int(np.ceil(n / rows))
    else:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    return max(1, rows), max(1, cols)


def create_small_multiples_layout(
    n_panels: int,
    *,
    nrows_override: Optional[int] = None,
    ncols_override: Optional[int] = None,
    include_legend_slot: bool = True,
    per_panel_size: float = OVERLAY_PER_PANEL_SIZE,
    panel_w_ratio: float = OVERLAY_PANEL_W_RATIO,
    panel_h_ratio: float = OVERLAY_PANEL_H_RATIO,
) -> SmallMultiplesLayout:
    """Build the matplotlib figure/axes grid used by small-multiple plots."""

    total_slots = n_panels + (1 if include_legend_slot else 0)
    total_slots = max(1, total_slots)

    nrows, ncols = determine_grid(total_slots, nrows_override, ncols_override)
    fig_w = per_panel_size * panel_w_ratio * ncols
    fig_h = per_panel_size * panel_h_ratio * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes_flat = np.atleast_2d(axes).reshape(-1)

    legend_ax = axes_flat[n_panels] if include_legend_slot and len(axes_flat) > n_panels else None
    extra_start = n_panels + (1 if legend_ax is not None else 0)
    extra_axes = list(axes_flat[extra_start:])
    panel_axes = axes_flat[:n_panels]

    positions = [(idx // ncols, idx % ncols) for idx in range(n_panels)]
    bottom_row_for_col: Dict[int, int] = {}
    for row, col in positions:
        bottom_row_for_col[col] = max(bottom_row_for_col.get(col, -1), row)

    return SmallMultiplesLayout(
        fig=fig,
        panel_axes=panel_axes,
        legend_ax=legend_ax,
        extra_axes=extra_axes,
        positions=positions,
        bottom_row_for_col=bottom_row_for_col,
        nrows=nrows,
        ncols=ncols,
    )


def _even_ticks_from_spec(
    ticks: Sequence[float],
    minimum: float,
    maximum: float,
) -> List[float]:
    """Return even integer ticks within the provided bounds."""

    tolerance = 1e-8
    even_ticks: List[float] = []
    integer_values: List[int] = []
    has_non_integer = False
    for tick in ticks:
        try:
            value = float(tick)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        rounded = round(value)
        if abs(value - rounded) <= tolerance:
            integer_values.append(int(rounded))
            if rounded % 2 == 0:
                even_ticks.append(float(rounded))
        else:
            has_non_integer = True

    if has_non_integer:
        return [float(tick) for tick in ticks]

    if even_ticks:
        return even_ticks

    if not (math.isfinite(minimum) and math.isfinite(maximum)):
        return even_ticks

    if integer_values:
        min_val = min(integer_values)
        max_val = max(integer_values)
    else:
        min_val = minimum
        max_val = maximum

    start = math.ceil(min_val)
    if start % 2 != 0:
        start += 1
    candidate = float(start)
    upper_bound = max(max_val, maximum)
    while candidate <= upper_bound + tolerance:
        even_ticks.append(candidate)
        candidate += 2.0

    if not even_ticks and minimum <= 0.0 <= maximum:
        even_ticks.append(0.0)

    return even_ticks


def apply_axis_spec(ax: plt.Axes, spec: AxisSpec) -> None:
    """Apply the shared axis limits and ticks with optional padding."""

    x_pad = spec.x_buffer_frac * (spec.x_max - spec.x_min)
    y_pad = spec.y_buffer_frac * (spec.y_max - spec.y_min)
    ax.set_xlim(spec.x_min - x_pad, spec.x_max + x_pad)
    ax.set_ylim(spec.y_min - y_pad, spec.y_max + y_pad)
    ax.set_xticks(_even_ticks_from_spec(spec.x_ticks, spec.x_min, spec.x_max))
    ax.set_yticks(_even_ticks_from_spec(spec.y_ticks, spec.y_min, spec.y_max))


def apply_titles_and_layout(layout: SmallMultiplesLayout, title_spec: TitleSpec) -> None:
    """Attach shared titles/labels and tighten the layout."""

    if title_spec.title:
        layout.fig.suptitle(title_spec.title, fontsize=title_spec.title_fontsize, y=title_spec.title_y)
    if title_spec.xlabel:
        layout.fig.supxlabel(title_spec.xlabel, fontsize=title_spec.xlabel_fontsize, y=title_spec.xlabel_y)
    if title_spec.ylabel:
        layout.fig.supylabel(title_spec.ylabel, fontsize=title_spec.ylabel_fontsize, x=title_spec.ylabel_x)
    tight_kwargs = {"pad": title_spec.tight_layout_pad}
    if title_spec.tight_layout_w_pad is not None:
        tight_kwargs["w_pad"] = title_spec.tight_layout_w_pad
    if title_spec.tight_layout_h_pad is not None:
        tight_kwargs["h_pad"] = title_spec.tight_layout_h_pad
    layout.fig.tight_layout(rect=title_spec.tight_layout_rect, **tight_kwargs)

    adjust_kwargs = {}
    if title_spec.subplot_wspace is not None:
        adjust_kwargs["wspace"] = title_spec.subplot_wspace
    if title_spec.subplot_hspace is not None:
        adjust_kwargs["hspace"] = title_spec.subplot_hspace
    if adjust_kwargs:
        layout.fig.subplots_adjust(**adjust_kwargs)


def hide_extra_axes(layout: SmallMultiplesLayout) -> None:
    """Turn off any unused axes slots (legend already handled separately)."""

    for ax in layout.extra_axes:
        ax.axis('off')

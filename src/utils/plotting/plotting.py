#!/usr/bin/env python3
from __future__ import annotations

import random
import math
from typing import Tuple, Optional

import matplotlib.pyplot as plt

from src.utils.plotting.figure_naming import figure_output_stem

DEFAULT_FIGURE_DPI = 300
# Keep publication exports consistent when savefig() is called without an explicit dpi.
plt.rcParams["savefig.dpi"] = DEFAULT_FIGURE_DPI

# GLOBAL PARAMETERS (plot styling)
FIG_SIZE = 5.0
MARKER_SIZE = 150.0
MARKER_ALPHA = 0.5
REG_FILL_ALPHA = 0.2
MARKER_COLOR = "steelblue"
REG_LINE_WIDTH = 3.5
REG_COLOR = "darkorange"

# Type-specific colors for overlay grid
TYPE_X_COLOR = "#1f77b4"  # blue     (Classic)
TYPE_Y_COLOR = "#ff7f0e"  # orange   (Adaptive)
CH_CLASSIC_LABEL = "Classic"
CH_ADAPTIVE_LABEL = "Adaptive"

# Correlation plot defaults
CORR_EQUAL_AXES_DEFAULT = True
CORR_FIXED_LIMITS_DEFAULT: Optional[Tuple[float, float]] = (5.0, 17.0)
CORR_JITTER_CLOSE_THRESH = 0.25
CORR_JITTER_MAX_JITTER = 0.1
CORR_DEFAULT_TITLE = "Paired $\\psi_{\\theta}$"
CORR_DEFAULT_X_LABEL = "$\\psi_{\\theta}$ (Classic)"
CORR_DEFAULT_Y_LABEL = "$\\psi_{\\theta}$ (Adaptive)"
CORR_DEFAULT_FILENAME_STEM = figure_output_stem("Fig04")
CORR_SHOW_PID_LABELS_DEFAULT = True
CORR_PID_LABEL_SUFFIX_DIGITS = 3
CORR_LABEL_PLACEMENT_ORDER: Tuple[Tuple[str, Tuple[int, int]], ...] = (
    ("left", (-9, 0)),
    ("below", (0, -9)),
    ("above", (0, 9)),
)
CORR_LABEL_CLASH_THRESHOLD = 0.25

# Correlation outlier handling
CORR_EXCLUDE_OUTLIERS_DEFAULT = True
CORR_OUTLIER_METHOD_DEFAULT = "iqr_diff"
CORR_IQR_MULTIPLIER_DEFAULT = 1.5
CORR_ZSCORE_THRESHOLD_DEFAULT = 3.0

AXIS_LABEL_FONT_SIZE = 14
TICK_LABEL_FONT_SIZE = 16
SUPER_TITLE_FONT_SIZE = 14

N_MAJOR_TICKS = 5
TICK_LENGTH = 4
TICK_WIDTH = 1
SPINE_LINEWIDTH = 1

AXIS_BUFFER_RATIO = 0.05
DECIMALS_FOR_TICKS = 1

random.seed(42)

# Overlay grid axes and bars
OVERLAY_X_MIN = 2.0
OVERLAY_X_MAX = 16.0
OVERLAY_X_TICK_STEP = 2
OVERLAY_POINT_SIZE = 40
OVERLAY_POINT_ALPHA = 0.8
OVERLAY_LINE_WIDTH = 1.8
OVERLAY_PER_PANEL_SIZE = 2.6
OVERLAY_PANEL_W_RATIO = 1.25
OVERLAY_PANEL_H_RATIO = 1.00
OVERLAY_SUBPLOT_WSPACE = 0.17
OVERLAY_SUBPLOT_HSPACE = 0.40
SIGMA_BAR_Y_X = 52
SIGMA_BAR_Y_Y = 48
OVERLAY_GRID_TITLE = "Classic Mode Sigmoids vs Adaptive Mode Posteriors Evaluated at Color = 3"

OVERLAY_BOTTOM_ROW_XLABELS_ONLY = True
OVERLAY_LEFT_COL_YLABELS_ONLY = True

SUPTITLE_Y = 0.98
TIGHT_RECT = (0.03, 0.02, 1.0, 0.95)
TIGHT_PAD = 0.4
TIGHT_W_PAD = 0.3
TIGHT_H_PAD = 0.3
SUPER_X_LABEL = "Difficulty (Block Count)"
SUPER_Y_LABEL = "Probability Correct (%)"
SUPER_Y_LABEL_X = 0.01

SPINES_BOTTOM_LEFT_ONLY = True

GLOBAL_LEGEND_LOC = 'lower right'
GLOBAL_LEGEND_IN_LAST_AXIS = False

# Subplot-specific sizing (overlay grid)
SUBPLOT_TICK_LABEL_FONT_SIZE_X = 11
SUBPLOT_TICK_LABEL_FONT_SIZE_Y = 11
SUBPLOT_TITLE_FONT_SIZE = 12
SUBPLOT_LEGEND_LOC = 'upper right'
SUBPLOT_LEGEND_FONTSIZE = 8

FIT_INIT_GUESS: Tuple[float, float] = (5.0, 1.0)
FIT_BOUNDS_X0: Tuple[float, float] = (0.0, 20.0)
FIT_BOUNDS_SPREAD: Tuple[float, float] = (0.1, 10.0)
FIT_CAP_X0: Optional[float] = None
FIT_MAXFEV = 5000

AXIS_VIEW_BUFFER_FRAC = 0.01


def _style_spines_and_ticks(ax):
    if SPINES_BOTTOM_LEFT_ONLY:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(SPINE_LINEWIDTH)
            ax.spines[spine].set_color("black")
        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=TICK_LENGTH,
            width=TICK_WIDTH,
            colors="black",
            labelsize=TICK_LABEL_FONT_SIZE,
            bottom=True, top=False, left=True, right=False,
        )
    else:
        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(SPINE_LINEWIDTH)
            ax.spines[spine].set_color("black")
        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=TICK_LENGTH,
            width=TICK_WIDTH,
            colors="black",
            labelsize=TICK_LABEL_FONT_SIZE,
        )

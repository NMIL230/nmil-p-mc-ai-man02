from __future__ import annotations

"""Shared helpers for canonical manuscript figure naming and output paths."""

from pathlib import Path
from typing import Dict

FIGURE_NAME_SETS: Dict[str, str] = {
    "Fig02": "Figure02_Patterns",
    "Fig03": "Figure03_Samples&Contours",
    "Fig04": "Figure04_Correlation",
    "Fig05": "Figure05_OrderEffects",
    "Fig06": "Figure06_AccuracyEvolution",
    "Fig07": "Figure07_ContourAccuracy",
}

MANUSCRIPT_FIGURE_ROOT = Path("manuscript_figures")


def figure_display_name(key: str) -> str:
    """Return the canonical display name for a figure key."""
    return FIGURE_NAME_SETS.get(key, key)


def figure_output_dir(key: str, root: Path | None = None) -> Path:
    """Return the default output directory for a figure key."""
    base = root if root is not None else MANUSCRIPT_FIGURE_ROOT
    return Path(base) / figure_display_name(key)


def figure_output_stem(key: str) -> str:
    """Return the default filename stem for a figure key."""
    return figure_display_name(key)

"""Shared helpers to load AMLEC data and construct fits/pairs for downstream analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.data.amlec_loader import load_amlec_diffmaps
from src.utils.modeling.fits import fit_sigmoids_per_entity
from src.utils.data.pairing import filter_valid_psi_pairs, pair_psis_with_entities

__all__ = ["build_fits_and_pairs"]


def build_fits_and_pairs(
    pickle_path: Path | None = None,
) -> Tuple[
    Dict[str, List[Dict[str, float]]],  # fits_x
    Dict[str, List[Dict[str, float]]],  # fits_y
    List[Tuple[str, float, float]],  # ent_pairs
    List[Tuple[float, float]],  # valid_pairs
    List[Tuple[str, float, float]],  # valid_ent_pairs
    List[str],  # point_labels
    Dict[str, List[dict]],  # gp_by_pid
    Dict[str, List[Tuple[float, float, int]]],  # adaptive_raw_points_all
]:
    """Load dataset, fit sigmoids, pair psi values and filter for validity."""

    if pickle_path is None:
        data_x, data_y, _diag, adaptive_raw_scatter, adaptive_raw_points_all, gp_by_pid = load_amlec_diffmaps()
    else:
        data_x, data_y, _diag, adaptive_raw_scatter, adaptive_raw_points_all, gp_by_pid = load_amlec_diffmaps(
            Path(pickle_path)
        )

    fits_x = fit_sigmoids_per_entity(
        data_x,
        fit_kwargs=dict(cap_x0=18.0),
        phantom_failure_at=17.0,
    )
    fits_y = fit_sigmoids_per_entity(data_y)
    ent_pairs = pair_psis_with_entities(fits_x, fits_y, mode="first_vs_first")
    pairs, point_labels, valid_ent_pairs, _report = filter_valid_psi_pairs(
        ent_pairs,
        min_allowed=0.0,
        max_allowed=18.0,
    )
    return fits_x, fits_y, ent_pairs, pairs, valid_ent_pairs, point_labels, gp_by_pid, adaptive_raw_points_all

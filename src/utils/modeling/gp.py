#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from typing import Tuple, List


def get_axes_from_grid(grid_xx: np.ndarray, grid_yy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(grid_xx[0, :], dtype=float)
    y_values = np.asarray(grid_yy[:, 0], dtype=float)
    return x_values, y_values


def slice_posterior_color3(pkl_data: dict) -> Tuple[np.ndarray, np.ndarray]:
    grid_xx = pkl_data["grid_xx"]
    grid_yy = pkl_data["grid_yy"]
    zz = pkl_data["zz_posterior"]
    if zz.ndim == 1:
        zz = zz.reshape(grid_xx.shape)
    x_vals, y_vals = get_axes_from_grid(grid_xx, grid_yy)
    color_idx = int(np.argmin(np.abs(y_vals - 3.0)))
    posterior_curve = zz[color_idx, :].astype(float)
    block_values = x_vals.astype(float)
    return block_values, posterior_curve


def interpolate_to_common_blocks(curves: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    min_b = int(min(np.min(b) for b, _ in curves))
    max_b = int(max(np.max(b) for b, _ in curves))
    common_blocks = np.arange(min_b, max_b + 1, dtype=float)
    interp_curves = []
    for b, p in curves:
        interp = np.interp(common_blocks, b.astype(float), p.astype(float))
        interp_curves.append(interp)
    mean_curve = np.mean(np.vstack(interp_curves), axis=0)
    return common_blocks, mean_curve


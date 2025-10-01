#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


# Fitting defaults mirrored from analysis module
FIT_INIT_GUESS: Tuple[float, float] = (5.0, 1.0)
FIT_BOUNDS_X0: Tuple[float, float] = (0.0, 20.0)
FIT_BOUNDS_SPREAD: Tuple[float, float] = (0.1, 10.0)
FIT_CAP_X0: Optional[float] = 17.0
FIT_MAXFEV = 5000

# Guess and lapse used in logistic
FIT_GUESS: float = 0.04
FIT_LAPSE: float = 0.04


def logistic(x, x0, spread):
    k = 2 * np.log(3) / spread
    g = float(FIT_GUESS)
    l = float(FIT_LAPSE)
    return 100.0 * (g + (1.0 - g - l) / (1.0 + np.exp(k * (x - x0))))


def fit_sigmoid_from_diff_map(
    diff_map: Dict[float, Dict[str, float]],
    init_guess: Tuple[float, float] = FIT_INIT_GUESS,
    bounds_x0: Tuple[float, float] = FIT_BOUNDS_X0,
    bounds_spread: Tuple[float, float] = FIT_BOUNDS_SPREAD,
    cap_x0: Optional[float] = FIT_CAP_X0,
    maxfev: int = FIT_MAXFEV,
) -> Optional[Dict[str, float]]:
    if diff_map is None or len(diff_map) < 2:
        return None
    sorted_items = sorted(diff_map.items(), key=lambda kv: kv[0])
    x_data = np.array([float(k) for k, _ in sorted_items], dtype=float)
    y_data = np.array([
        (entry.get("successes", 0.0) / max(1.0, entry.get("trials", 0.0))) * 100.0
        for _, entry in sorted_items
    ])
    lower_bounds = [bounds_x0[0], bounds_spread[0]]
    upper_bounds = [bounds_x0[1], bounds_spread[1]]
    try:
        popt, _ = curve_fit(
            logistic, x_data, y_data,
            p0=list(init_guess),
            bounds=(lower_bounds, upper_bounds),
            maxfev=maxfev,
        )
    except Exception:
        try:
            init_x0 = float(np.interp(50, np.flip(y_data), np.flip(x_data)))
            init_x0 = float(np.clip(init_x0, bounds_x0[0], bounds_x0[1]))
            popt, _ = curve_fit(
                logistic, x_data, y_data,
                p0=[init_x0, max(bounds_spread[0], 2.0)],
                bounds=(lower_bounds, upper_bounds),
                maxfev=maxfev,
            )
        except Exception:
            init_x0 = float(np.interp(50, np.flip(y_data), np.flip(x_data)))
            x0 = float(np.clip(init_x0, bounds_x0[0], bounds_x0[1]))
            if cap_x0 is not None:
                x0 = min(x0, cap_x0)
            rmse = float(np.sqrt(np.mean((y_data - x0) ** 2)))
            return {"psi_theta": x0, "spread": 2.0, "rmse": rmse}
    x0, spread = float(popt[0]), float(popt[1])
    if cap_x0 is not None:
        x0 = min(x0, float(cap_x0))
    y_fit = logistic(x_data, x0, spread)
    rmse = float(np.sqrt(np.mean((y_data - y_fit) ** 2)))
    return {"psi_theta": x0, "spread": spread, "rmse": rmse}


def fit_sigmoids_per_entity(
    data_by_entity: Dict[str, List[Dict[float, Dict[str, float]]]],
    fit_kwargs: Optional[dict] = None,
    *,
    phantom_failure_at: Optional[float] = None,
) -> Dict[str, List[Dict[str, float]]]:
    if fit_kwargs is None:
        fit_kwargs = {}
    out: Dict[str, List[Dict[str, float]]] = {}
    for entity, maps in data_by_entity.items():
        fits: List[Dict[str, float]] = []
        for dm in maps:
            dm_input = dm
            if phantom_failure_at is not None:
                phantom_key = float(phantom_failure_at)
                dm_aug: Dict[float, Dict[str, float]] = {}
                for key, val in dm.items():
                    try:
                        float_key = float(key)
                    except Exception:
                        float_key = key
                    if isinstance(val, dict):
                        dm_aug[float_key] = dict(val)
                    else:
                        dm_aug[float_key] = {"successes": float(val), "trials": 1.0}
                existing = dm_aug.get(phantom_key)
                if existing is None:
                    dm_aug[phantom_key] = {"successes": 0.0, "trials": 1.0}
                else:
                    successes = float(existing.get("successes", 0.0))
                    trials = float(existing.get("trials", 0.0))
                    trials = max(trials, 1.0) + 1.0
                    successes = min(successes, max(trials - 1.0, 0.0))
                    dm_aug[phantom_key] = {"successes": successes, "trials": trials}
                dm_input = dm_aug
            res = fit_sigmoid_from_diff_map(dm_input, **fit_kwargs)
            if res is not None:
                with_data = dict(res)
                with_data["data"] = dm
                fits.append(with_data)
        if fits:
            out[entity] = fits
    return out

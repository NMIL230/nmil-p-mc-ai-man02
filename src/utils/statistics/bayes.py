"""Utilities for Bayesian calculations sourced from R's BayesFactor package."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = ["compute_jzs_bf_corr", "compute_jzs_bf_one_sample", "bayesfactor_available"]

_BAYESFACTOR = None
_RPY2_ROBJECTS = None
_RPY2_READY = False
_RPY2_ERROR: Exception | None = None


def _detect_r_home() -> Path | None:
    r_home_env = os.environ.get("R_HOME")
    if r_home_env:
        return Path(r_home_env)
    try:
        output = subprocess.check_output(["R", "RHOME"], stderr=subprocess.STDOUT, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    path = Path(output.strip())
    if path.exists():
        os.environ["R_HOME"] = str(path)
        return path
    return None


def _prepend_env_path(var: str, path: str) -> None:
    if not path:
        return
    current = os.environ.get(var)
    if current:
        entries = current.split(os.pathsep)
        if path in entries:
            return
        os.environ[var] = os.pathsep.join([path, current])
    else:
        os.environ[var] = path


def _initialise_rpy2() -> None:
    global _BAYESFACTOR, _RPY2_ROBJECTS, _RPY2_READY, _RPY2_ERROR
    if _RPY2_READY or _RPY2_ERROR is not None:
        return

    r_home = _detect_r_home()
    if not r_home:
        _RPY2_ERROR = RuntimeError("Could not determine R_HOME; is R installed and on PATH?")
        return

    lib_dir = r_home / "lib"
    if sys.platform == "darwin":
        _prepend_env_path("DYLD_LIBRARY_PATH", str(lib_dir))
    elif sys.platform.startswith("linux"):
        _prepend_env_path("LD_LIBRARY_PATH", str(lib_dir))

    try:
        import rpy2.robjects as ro  # type: ignore[import]
        from rpy2.robjects.packages import importr  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        _RPY2_ERROR = exc
        return

    try:
        _BAYESFACTOR = importr("BayesFactor")
    except Exception as exc:  # noqa: BLE001
        _RPY2_ERROR = exc
        return

    _RPY2_ROBJECTS = ro
    _RPY2_READY = True


def bayesfactor_available() -> bool:
    """Return True if the BayesFactor R package is available via rpy2."""
    _initialise_rpy2()
    return _RPY2_READY and _BAYESFACTOR is not None


def compute_jzs_bf_corr(
    x: Sequence[float],
    y: Sequence[float],
    rscale: float = 1.0 / math.sqrt(2.0),
) -> float:
    """Compute the Jeffreys-Zellner-Siow Bayes factor for a correlation."""
    _initialise_rpy2()
    if not (_RPY2_READY and _BAYESFACTOR is not None and _RPY2_ROBJECTS is not None):
        if _RPY2_ERROR is not None:
            raise RuntimeError("BayesFactor via rpy2 is unavailable") from _RPY2_ERROR
        raise RuntimeError("BayesFactor via rpy2 is unavailable")

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have identical shapes for paired Bayes factor computation")
    if x_arr.size < 3:
        raise ValueError("At least three paired observations are required for a correlation Bayes factor")

    r_vector_x = _RPY2_ROBJECTS.FloatVector(x_arr.tolist())
    r_vector_y = _RPY2_ROBJECTS.FloatVector(y_arr.tolist())
    result = _BAYESFACTOR.correlationBF(y=r_vector_y, x=r_vector_x, rscale=rscale)
    bf_table = result.slots["bayesFactor"]
    log_bf_vector = bf_table.rx2("bf")
    log_bf = float(log_bf_vector[0])
    if not math.isfinite(log_bf):
        raise RuntimeError("BayesFactor returned a non-finite log Bayes factor")
    return math.exp(log_bf)


def compute_jzs_bf_one_sample(
    values: Sequence[float],
    *,
    mu: float = 0.0,
    rscale: float = 1.0 / math.sqrt(2.0),
) -> float:
    """Compute a one-sample (paired-difference) Bayes factor via JZS prior."""
    _initialise_rpy2()
    if not (_RPY2_READY and _BAYESFACTOR is not None and _RPY2_ROBJECTS is not None):
        if _RPY2_ERROR is not None:
            raise RuntimeError("BayesFactor via rpy2 is unavailable") from _RPY2_ERROR
        raise RuntimeError("BayesFactor via rpy2 is unavailable")

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        raise ValueError("At least two observations are required for a one-sample Bayes factor")

    r_vector = _RPY2_ROBJECTS.FloatVector(arr.tolist())
    result = _BAYESFACTOR.ttestBF(x=r_vector, mu=mu, rscale=rscale)
    bf_table = result.slots["bayesFactor"]
    log_bf_vector = bf_table.rx2("bf")
    log_bf = float(log_bf_vector[0])
    if not math.isfinite(log_bf):
        raise RuntimeError("BayesFactor returned a non-finite log Bayes factor for the t-test")
    return math.exp(log_bf)

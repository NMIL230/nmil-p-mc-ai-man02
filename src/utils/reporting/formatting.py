#!/usr/bin/env python3
"""Formatting helpers enforcing consistent significant-figure and notation rules."""

from __future__ import annotations

import math
from typing import Optional


Number = Optional[float]


def _round_to_sigfigs(value: float, sigfigs: int) -> float:
    if sigfigs <= 0:
        raise ValueError("sigfigs must be positive")
    if value == 0.0:
        return 0.0
    magnitude = math.floor(math.log10(abs(value)))
    factor = 10 ** (sigfigs - 1 - magnitude)
    return round(value * factor) / factor


def _count_sigfigs(text: str) -> int:
    main = text.split("e")[0].split("E")[0]
    digits = 0
    seen_non_zero = False
    for ch in main:
        if ch.isdigit():
            if ch != "0" or seen_non_zero:
                digits += 1
                seen_non_zero = True
            elif seen_non_zero:
                digits += 1
    return digits


def format_stat(
    value: Number,
    *,
    mode: str,
    sigfigs: int = 3,
    nan_text: str = "NA",
) -> str:
    """Format *value* with a fixed number of significant figures in a target mode.

    Parameters
    ----------
    value:
        Float-like input. ``None`` or non-finite numbers map to *nan_text*.
    mode:
        Either ``"decimal"`` or ``"scientific"``.
    sigfigs:
        Number of significant figures (default 3).
    nan_text:
        Placeholder string if value is not finite.
    """

    if value is None:
        return nan_text
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return nan_text

    if not math.isfinite(numeric):
        return nan_text

    if numeric == 0.0:
        if mode == "scientific":
            return f"0.{(sigfigs - 1) * '0'}e+00"
        zero_decimals = max(sigfigs - 1, 0)
        if zero_decimals == 0:
            return "0"
        return f"0.{zero_decimals * '0'}"

    rounded = _round_to_sigfigs(numeric, sigfigs)

    if mode == "decimal":
        magnitude = math.floor(math.log10(abs(rounded)))
        if magnitude >= sigfigs:
            return format_stat(numeric, mode="scientific", sigfigs=sigfigs, nan_text=nan_text)
        decimals = max(sigfigs - 1 - magnitude, 0)
        text = f"{rounded:.{decimals}f}"
        if text.startswith("."):
            text = "0" + text
        elif text.startswith("-."):
            text = text.replace("-.", "-0.", 1)
    elif mode == "scientific":
        exponent = math.floor(math.log10(abs(rounded)))
        mantissa = rounded / (10 ** exponent)
        mantissa = round(mantissa, sigfigs - 1)
        text = f"{mantissa:.{sigfigs - 1}f}e{exponent:+03d}"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if sigfigs > 0 and numeric != 0.0:
        digits = _count_sigfigs(text)
        if digits != sigfigs:
            raise ValueError(
                f"Formatted value '{text}' does not contain {sigfigs} significant figures"
            )

    return text


__all__ = ["format_stat"]

#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.utils.plotting.labels import pid_display_label
from src.utils.reporting.formatting import format_stat


def pair_psis_with_entities(
    fits_x: Dict[str, List[Dict[str, float]]],
    fits_y: Dict[str, List[Dict[str, float]]],
    mode: str = "first_vs_first",
) -> List[Tuple[str, float, float]]:
    pairs: List[Tuple[str, float, float]] = []
    shared = set(fits_x.keys()) & set(fits_y.keys())
    for ent in shared:
        px = [f["psi_theta"] for f in fits_x.get(ent, [])]
        py = [f["psi_theta"] for f in fits_y.get(ent, [])]
        if not px or not py:
            continue
        if mode == "first_vs_all":
            x0 = px[0]
            for yv in py:
                pairs.append((ent, x0, yv))
        elif mode == "first_vs_first":
            pairs.append((ent, px[0], py[0]))
        elif mode == "all_vs_all":
            for xv in px:
                for yv in py:
                    pairs.append((ent, xv, yv))
        else:
            raise ValueError(f"Unknown pairing mode: {mode}")
    return pairs


def pair_psis(
    fits_x: Dict[str, List[Dict[str, float]]],
    fits_y: Dict[str, List[Dict[str, float]]],
    mode: str = "first_vs_first",
) -> List[Tuple[float, float]]:
    ent_pairs = pair_psis_with_entities(fits_x, fits_y, mode=mode)
    return [(x, y) for _, x, y in ent_pairs]


def filter_valid_psi_pairs(
    ent_pairs: Iterable[Tuple[str, Any, Any]],
    *,
    min_allowed: float = 0.0,
    max_allowed: float = 16.0,
    label_fn: Optional[Callable[[str], Optional[str]]] = None,
    max_detail: int = 10,
) -> Tuple[List[Tuple[float, float]], List[str], List[Tuple[str, float, float]], Dict[str, Any]]:
    def resolve_label(pid: str) -> str:
        lbl = label_fn(pid) if label_fn is not None else None
        if lbl is None or str(lbl).strip() == "":
            lbl = pid_display_label(pid)
        return str(lbl)

    def assess(value: Any, channel: str) -> Tuple[Optional[float], Optional[str]]:
        if value is None:
            return None, f"{channel} psi_theta missing"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None, f"{channel} psi_theta not numeric"
        if not np.isfinite(numeric):
            return None, f"{channel} psi_theta non-finite"
        if numeric < min_allowed or numeric > max_allowed:
            value_fmt = format_stat(numeric, mode="decimal")
            min_fmt = format_stat(min_allowed, mode="decimal")
            max_fmt = format_stat(max_allowed, mode="decimal")
            return None, f"{channel} psi_theta {value_fmt} outside [{min_fmt}, {max_fmt}]"
        return float(numeric), None

    def safe_value(raw: Any, coerced: Optional[float]) -> Optional[float]:
        if coerced is not None:
            return coerced
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    valid_pairs: List[Tuple[float, float]] = []
    valid_ent_pairs: List[Tuple[str, float, float]] = []
    labels: List[str] = []
    skipped_examples: List[Dict[str, Any]] = []
    skipped_count = 0
    total = 0

    for ent, raw_x, raw_y in ent_pairs:
        total += 1
        label = resolve_label(ent)
        x_val, x_reason = assess(raw_x, "Classic")
        y_val, y_reason = assess(raw_y, "Adaptive")
        reasons = [r for r in (x_reason, y_reason) if r is not None]
        if reasons:
            skipped_count += 1
            if len(skipped_examples) < max_detail:
                skipped_examples.append(
                    {
                        "participant": str(ent),
                        "label": label,
                        "classic": safe_value(raw_x, x_val),
                        "adaptive": safe_value(raw_y, y_val),
                        "reasons": reasons,
                    }
                )
            continue
        assert x_val is not None and y_val is not None
        valid_pairs.append((x_val, y_val))
        valid_ent_pairs.append((str(ent), x_val, y_val))
        labels.append(label)

    report: Dict[str, Any] = {
        "range": {"min": float(min_allowed), "max": float(max_allowed)},
        "total_pairs": total,
        "kept_pairs": len(valid_pairs),
        "skipped_pairs": skipped_count,
    }
    if skipped_examples:
        report["skipped_examples"] = skipped_examples
        if skipped_count > len(skipped_examples):
            report["skipped_examples_truncated"] = True

    return valid_pairs, labels, valid_ent_pairs, report

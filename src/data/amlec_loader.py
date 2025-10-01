#!/usr/bin/env python3
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.data.paths import AMLEC_PICKLE_PATH
from src.utils.modeling.gp import slice_posterior_color3, interpolate_to_common_blocks


CH_CLASSIC_LABEL = "Classic"
CH_ADAPTIVE_LABEL = "Adaptive"

AMLEC_ENTITY_KEY = "adaptive_session_id"
AMLEC_DIFF_KEY = "block_count"
AMLEC_PASS_KEY = "passed"
AMLEC_CHANNEL_FIELD = "color_count"
AMLEC_ADAPTIVE_REQUIRED_COLOR = 3
AMLEC_VERBOSE = True
AMLEC_MAX_ENTITY_DETAIL = 10


def _iter_amlec_entries(obj):
    if isinstance(obj, list):
        for it in obj:
            mod = None
            if isinstance(it, dict):
                gr = it.get("game_result") or it
                gm = str((gr or {}).get("game_mode", "")).upper()
                if "ADAPTIVE" in gm:
                    mod = "ADAPTIVE"
                elif "CLASSIC" in gm or "D2_3COLOR" in gm:
                    mod = "CLASSIC"
            yield (it, mod)
        return
    if isinstance(obj, dict):
        for key, val in obj.items():
            if isinstance(val, dict) and ("CLASSIC" in val or "ADAPTIVE" in val):
                for modality in ("CLASSIC", "ADAPTIVE"):
                    sub = val.get(modality)
                    if isinstance(sub, dict):
                        prod = sub.get("product")
                        if isinstance(prod, dict):
                            prod = dict(prod)
                            prod.setdefault("participant_id", str(key))
                            yield (prod, modality)
                continue
            yield (val, None)
        return
    yield (obj, None)


def load_amlec_diffmaps(
    pickle_path: Path = AMLEC_PICKLE_PATH,
) -> Tuple[
    Dict[str, List[Dict[float, Dict[str, float]]]],
    Dict[str, List[Dict[float, Dict[str, float]]]],
    Dict[str, Any],
    Dict[str, Dict[float, Dict[str, float]]],
    Dict[str, List[Tuple[float, float, int]]],
    Dict[str, List[dict]],
]:
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    data_classic: Dict[str, List[Dict[float, Dict[str, float]]]] = {}
    data_adaptive: Dict[str, List[Dict[float, Dict[str, float]]]] = {}
    adaptive_raw_scatter: Dict[str, Dict[float, Dict[str, float]]] = {}
    adaptive_raw_points_all: Dict[str, List[Tuple[float, float, int]]] = {}
    diag: Dict[str, Any] = {
        "pickle_path": str(pickle_path),
        "top_type": type(data).__name__,
        "total_entries": 0,
        "entries_with_game_result": 0,
        "entries_with_game_record": 0,
        "entries_with_level_records": 0,
        "total_level_records": 0,
        "channel_counts": {CH_CLASSIC_LABEL: 0, CH_ADAPTIVE_LABEL: 0},
        "entities_raw": {CH_CLASSIC_LABEL: 0, CH_ADAPTIVE_LABEL: 0},
        "entities_qualified": {CH_CLASSIC_LABEL: 0, CH_ADAPTIVE_LABEL: 0},
        "per_entity_diff_counts": {CH_CLASSIC_LABEL: {}, CH_ADAPTIVE_LABEL: {}},
        "samples_no_level_records": [],
    }

    gp_by_pid: Dict[str, List[dict]] = {}
    if isinstance(data, dict):
        for pid, bundle in data.items():
            if not isinstance(bundle, dict):
                continue
            gp_obj = None
            if isinstance(bundle.get("ADAPTIVE"), dict):
                gp_obj = bundle["ADAPTIVE"].get("GP") or bundle["ADAPTIVE"].get("gp")
            if gp_obj is None:
                gp_obj = bundle.get("GP") or bundle.get("gp")
            if gp_obj is None:
                continue
            gp_list = gp_by_pid.setdefault(str(pid), [])
            if isinstance(gp_obj, list):
                gp_list.extend([g for g in gp_obj if isinstance(g, dict)])
            elif isinstance(gp_obj, dict):
                gp_list.append(gp_obj)

    for item, modality in _iter_amlec_entries(data):
        if not isinstance(item, dict):
            continue
        diag["total_entries"] += 1
        gr = item.get("game_result") or item
        if not isinstance(gr, dict):
            continue
        diag["entries_with_game_result"] += 1
        ent = (
            gr.get("participant_id")
            or item.get("participant_id")
            or gr.get(AMLEC_ENTITY_KEY)
            or item.get(AMLEC_ENTITY_KEY)
            or "UNKNOWN"
        )
        ent = str(ent)
        game_record = gr.get("game_record") or {}
        if isinstance(game_record, dict):
            diag["entries_with_game_record"] += 1
        level_records = game_record.get("level_records") or {}
        if not isinstance(level_records, dict) or not level_records:
            if len(diag["samples_no_level_records"]) < AMLEC_MAX_ENTITY_DETAIL:
                try:
                    diag["samples_no_level_records"].append({
                        "participant": ent,
                        "product_keys": sorted(list(gr.keys())),
                        "has_game_record": isinstance(game_record, dict),
                        "game_record_keys": sorted(list(game_record.keys())) if isinstance(game_record, dict) else None,
                        "level_records_type": type(level_records).__name__,
                        "level_records_len": len(level_records) if hasattr(level_records, '__len__') else None,
                    })
                except Exception:
                    pass
            continue
        if level_records:
            diag["entries_with_level_records"] += 1

        channel_label = None
        if isinstance(modality, str):
            mu = modality.upper()
            if mu == "CLASSIC":
                channel_label = CH_CLASSIC_LABEL
            elif mu == "ADAPTIVE":
                channel_label = CH_ADAPTIVE_LABEL

        for rec in level_records.values():
            if not isinstance(rec, dict):
                continue
            diff = rec.get(AMLEC_DIFF_KEY)
            if diff is None:
                continue
            passed = bool(rec.get(AMLEC_PASS_KEY))

            ch = channel_label
            if ch is None:
                gm = str(gr.get("game_mode", "")).upper()
                ch = CH_ADAPTIVE_LABEL if "ADAPTIVE" in gm else CH_CLASSIC_LABEL

            if ch == CH_CLASSIC_LABEL:
                target = data_classic
            else:
                cc_val = rec.get(AMLEC_CHANNEL_FIELD)
                try:
                    cc_float = float(cc_val)
                except Exception:
                    cc_float = None
                if cc_float is not None:
                    pts = adaptive_raw_points_all.setdefault(str(ent), [])
                    pts.append((float(diff), float(cc_float), int(1 if passed else 0)))

                try:
                    if int(cc_val) != int(AMLEC_ADAPTIVE_REQUIRED_COLOR):
                        continue
                except Exception:
                    continue
                dm = adaptive_raw_scatter.setdefault(str(ent), {})
                key = float(diff)
                if key not in dm:
                    dm[key] = {"successes": 0, "trials": 0}
                dm[key]["trials"] += 1
                if passed:
                    dm[key]["successes"] += 1
                diag["total_level_records"] += 1
                diag["channel_counts"][CH_ADAPTIVE_LABEL] += 1
                continue

            dm_list = target.setdefault(str(ent), [])
            if not dm_list:
                dm_list.append({})
            dm = dm_list[0]
            key = float(diff)
            if key not in dm:
                dm[key] = {"successes": 0, "trials": 0}
            dm[key]["trials"] += 1
            if passed:
                dm[key]["successes"] += 1
            diag["total_level_records"] += 1
            diag["channel_counts"][CH_CLASSIC_LABEL] += 1

    diag["entities_raw"][CH_CLASSIC_LABEL] = len(data_classic)
    # Post-process Classic: if subject has perfect passes through length 16, add synthetic fail at 17
    try:
        for pid, dm_list in list(data_classic.items()):
            if not dm_list:
                continue
            dm = dm_list[0]
            if not isinstance(dm, dict) or not dm:
                continue
            keys = sorted(dm.keys())
            # Consider trials up to and including 16
            up_to_16 = [k for k in keys if float(k) <= 16.0 + 1e-9]
            if not up_to_16:
                continue
            all_perfect = True
            for k in up_to_16:
                rec = dm.get(k) or {}
                tr = float(rec.get("trials", 0))
                sc = float(rec.get("successes", 0))
                if tr <= 0 or sc < tr:
                    all_perfect = False
                    break
            if all_perfect:
                if 17.0 not in dm:
                    dm[17.0] = {"successes": 0.0, "trials": 1.0}
                else:
                    # Add one synthetic failed trial at 17
                    try:
                        dm[17.0]["trials"] = float(dm[17.0].get("trials", 0.0)) + 1.0
                        # successes unchanged to reflect a fail
                    except Exception:
                        dm[17.0] = {"successes": float(dm[17.0].get("successes", 0.0)), "trials": float(dm[17.0].get("trials", 0.0)) + 1.0}
    except Exception:
        pass
    for pid, gp_list in gp_by_pid.items():
        curves: List[Tuple[np.ndarray, np.ndarray]] = []
        for gp in gp_list:
            try:
                bvals, curve = slice_posterior_color3(gp)
                curves.append((bvals, curve))
            except Exception:
                continue
        if curves:
            blocks, mean_curve = interpolate_to_common_blocks(curves)
            dm: Dict[float, Dict[str, float]] = {}
            for b, p in zip(blocks, mean_curve):
                dm[float(b)] = {"successes": float(p), "trials": 1.0}
            data_adaptive.setdefault(str(pid), []).append(dm)

    # Fallback: if no GP-derived adaptive data for a PID, use raw adaptive scatter at color_count==3
    for pid, dm in adaptive_raw_scatter.items():
        if str(pid) not in data_adaptive and isinstance(dm, dict) and len(dm) >= 2:
            data_adaptive[str(pid)] = [dm]

    for e, lst in data_classic.items():
        if not lst:
            continue
        dm = lst[0]
        if len(dm) >= 2:
            diag["per_entity_diff_counts"][CH_CLASSIC_LABEL][e] = len(dm)
    for e, lst in data_adaptive.items():
        if not lst:
            continue
        dm = lst[0]
        if len(dm) >= 2:
            diag["per_entity_diff_counts"][CH_ADAPTIVE_LABEL][e] = len(dm)

    data_classic = {e: [dm] for e, (dm,) in ((e, lst) for e, lst in data_classic.items() if lst) if len(dm) >= 2}
    data_adaptive = {e: [dm] for e, (dm,) in ((e, lst) for e, lst in data_adaptive.items() if lst) if len(dm) >= 2}
    diag["entities_qualified"][CH_CLASSIC_LABEL] = len(data_classic)
    diag["entities_qualified"][CH_ADAPTIVE_LABEL] = len(data_adaptive)

    return data_classic, data_adaptive, diag, adaptive_raw_scatter, adaptive_raw_points_all, gp_by_pid

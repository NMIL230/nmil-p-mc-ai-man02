#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.data.paths import PROJECT_ROOT, AMLEC_PICKLE_PATH


def _parse_game_start(ts: object) -> Optional[datetime]:
    if not isinstance(ts, str) or not ts:
        return None
    t = ts.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(t)
    except Exception:
        return None


def determine_mode_order_for_pid(pid: str, pid_base_dir: Optional[Path] = None) -> Optional[str]:
    """Return 'classic_first' or 'adaptive_first' based on earliest start timestamps.

    Looks for JSON files under data/PID/<pid> with game_result.game_mode and game_start_time.
    """
    base = pid_base_dir or (PROJECT_ROOT / "data" / "PID")
    pid_dir = Path(base) / str(pid)
    if not pid_dir.exists() or not pid_dir.is_dir():
        return _determine_mode_order_from_dataset(pid)
    classic_times: List[datetime] = []
    adaptive_times: List[datetime] = []
    try:
        for p in pid_dir.glob("*.json"):
            if p.name.startswith("._"):
                continue
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            st = data.get("game_start_time") if isinstance(data, dict) else None
            gr = data.get("game_result") if isinstance(data, dict) else None
            if not isinstance(gr, dict):
                gr = {}
            mode = str(gr.get("game_mode", "")).upper()
            ts = _parse_game_start(st)
            if ts is None:
                continue
            if mode == "D2_3COLOR":
                classic_times.append(ts)
            elif "ADAPTIVE" in mode:
                adaptive_times.append(ts)
    except Exception:
        return None
    if not classic_times or not adaptive_times:
        if classic_times or adaptive_times:
            # Only one modality recorded in raw logs – fall back to dataset for missing start time.
            return _determine_mode_order_from_dataset(pid, classic_times, adaptive_times)
        return _determine_mode_order_from_dataset(pid)
    c_min = min(classic_times)
    a_min = min(adaptive_times)
    if a_min < c_min:
        return "adaptive_first"
    if c_min < a_min:
        return "classic_first"
    return None


_DATASET_CACHE: Dict[str, Any] | None = None


def _load_dataset_cache() -> Dict[str, Any]:
    global _DATASET_CACHE
    if _DATASET_CACHE is None:
        try:
            with AMLEC_PICKLE_PATH.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        _DATASET_CACHE = payload
    return _DATASET_CACHE


def _determine_mode_order_from_dataset(
    pid: str,
    classic_times: Optional[List[datetime]] = None,
    adaptive_times: Optional[List[datetime]] = None,
) -> Optional[str]:
    dataset = _load_dataset_cache()
    entry = dataset.get(pid)
    if not isinstance(entry, dict):
        return None

    if classic_times is None:
        classic_times = []
    if adaptive_times is None:
        adaptive_times = []

    def _extract_start(modality_key: str) -> Optional[datetime]:
        bundle = entry.get(modality_key)
        if not isinstance(bundle, dict):
            return None
        process = bundle.get("process")
        if isinstance(process, dict):
            ts_candidates: List[datetime] = []
            for trial in process.values():
                if not isinstance(trial, dict):
                    continue
                for phase_key in ("OBSERVE_PHASE", "BUILD_PHASE"):
                    phase = trial.get(phase_key)
                    if not isinstance(phase, dict):
                        continue
                    stamp = _parse_game_start(phase.get("start_time"))
                    if stamp is not None:
                        ts_candidates.append(stamp)
                        break
                if ts_candidates:
                    break
            if ts_candidates:
                return min(ts_candidates)
        product = bundle.get("product")
        if isinstance(product, dict):
            game_record = product.get("game_record")
            if isinstance(game_record, dict):
                started = game_record.get("start_time") or game_record.get("started_at")
                parsed = _parse_game_start(started)
                if parsed is not None:
                    return parsed
        return None

    if not classic_times:
        classic_start = _extract_start("CLASSIC")
        if classic_start is not None:
            classic_times = [classic_start]
    if not adaptive_times:
        adaptive_start = _extract_start("ADAPTIVE")
        if adaptive_start is not None:
            adaptive_times = [adaptive_start]

    if not classic_times or not adaptive_times:
        return None

    c_min = min(classic_times)
    a_min = min(adaptive_times)
    if a_min < c_min:
        return "adaptive_first"
    if c_min < a_min:
        return "classic_first"
    return None

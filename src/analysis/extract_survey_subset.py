#!/usr/bin/env python3
"""Extract a survey subset into a real table (DataFrame/CSV).

Selects only the following questions by their human-readable labels
from a Qualtrics CSV export (with three header rows):

- 0.1: Subject ID
- 7.4: Approaches (free response)
- 7.5: Fatigue A
- 7.6: Fatigue C

Outputs a tidy table with columns: pid, mouse_keyboard_comfort, familiarity, approach, fatigue_a, fatigue_c.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data.pids import normalize_pid

try:  # Optional dependency for DataFrame return
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas not installed
    pd = None  # type: ignore


DEFAULT_CSV_PATH = Path(
    "data/survey/Active+Machine+Learning+for+Evaluating+Cognition_September+20,+2025_10.28.csv"
)
DEFAULT_OUTPUT_CSV = Path("results/survey_subset.csv")
DEFAULT_PICKLE_PATH = Path("data/survey/survey_subset.pkl")


# Human-readable column labels in the Qualtrics export (second header row)
PID_COLUMN_LABEL = "Subject ID"  # 0.1
APPROACH_COLUMN_LABEL = (
    "If you can describe the approach(es) you used to play the game successfully, "
    "please write about that here. Importantly, indicate any differences in approach "
    "between the two modes."
)
FATIGUE_A_COLUMN_LABEL = "At the end of Mode A, how mentally fatigued did you feel? - 1"  # 7.5_1
FATIGUE_C_COLUMN_LABEL = "At the end of Mode C, how mentally fatigued did you feel? - 1"  # 7.6_1
MOUSE_KEYBOARD_COMFORT_COLUMN_LABEL = (
    "How would you rate your comfort with using mouse and keyboard for gaming? - Comfort with Mouse and Keyboard"
)
FAMILIARITY_COLUMN_LABEL = "How would you rate your familiarity with Minecraft? - Familiarity"  # 3.1_1
SEX_COLUMN_LABEL = "What is your sex?"
AGE_COLUMN_LABEL = "As of today, how old are you (in years)?"
DEGREE_COLUMN_LABEL = (
    "What is the highest level of school you have completed or the highest degree you have received?"
)
HISPANIC_COLUMN_LABEL = "Are you Spanish, Hispanic, or Latino or none of these?"
RACE_COLUMN_LABEL = "Choose one or more races that you consider yourself to be: - Selected Choice"
ENJOY_C_COLUMN_LABEL = "How much did you enjoy each game? - Build Master Mode C"
ENJOY_A_COLUMN_LABEL = "How much did you enjoy each game? - Build Master Mode A"
FINISHED_COLUMN_LABEL = "Finished"


def _read_csv_rows(csv_path: Path) -> Tuple[Sequence[str], Sequence[str], Sequence[str], List[Sequence[str]]]:
    """Load the Qualtrics CSV and return header rows + data rows."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            row_short = next(reader)
            row_labels = next(reader)
            row_meta = next(reader)
        except StopIteration as exc:  # pragma: no cover - malformed input
            raise ValueError("CSV appears to be missing standard Qualtrics header rows") from exc
        data_rows = [row for row in reader if row]
    return row_short, row_labels, row_meta, data_rows


def _find_column(label_row: Sequence[str], target: str) -> int:
    labels = [col.strip() if isinstance(col, str) else col for col in label_row]
    try:
        return labels.index(target)
    except ValueError as exc:
        raise ValueError(f"Unable to locate column: {target!r}") from exc


def _resolve_csv_path(csv_path: Path) -> Path:
    if csv_path.exists():
        return csv_path
    if csv_path == DEFAULT_CSV_PATH:
        survey_dir = DEFAULT_CSV_PATH.parent
        candidates = sorted(survey_dir.glob("Active+Machine+Learning+for+Evaluating+Cognition_*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No survey CSV found in {survey_dir}")
        return candidates[-1]
    raise FileNotFoundError(f"CSV not found: {csv_path}")


def extract_records(csv_path: Path) -> List[Dict[str, str]]:
    """Return a list of dict records with pid, approach, fatigue_a, fatigue_c."""
    _, label_row, _meta_row, data_rows = _read_csv_rows(csv_path)

    pid_idx = _find_column(label_row, PID_COLUMN_LABEL)
    approach_idx = _find_column(label_row, APPROACH_COLUMN_LABEL)
    fatigue_a_idx = _find_column(label_row, FATIGUE_A_COLUMN_LABEL)
    fatigue_c_idx = _find_column(label_row, FATIGUE_C_COLUMN_LABEL)
    try:
        familiarity_idx = _find_column(label_row, FAMILIARITY_COLUMN_LABEL)
    except ValueError:
        familiarity_idx = None
    try:
        mouse_keyboard_comfort_idx = _find_column(label_row, MOUSE_KEYBOARD_COMFORT_COLUMN_LABEL)
    except ValueError:
        mouse_keyboard_comfort_idx = None

    records: List[Dict[str, str]] = []
    for row in data_rows:
        # guard against shorter rows
        idxs = [pid_idx, approach_idx, fatigue_a_idx, fatigue_c_idx]
        if familiarity_idx is not None:
            idxs.append(familiarity_idx)
        if mouse_keyboard_comfort_idx is not None:
            idxs.append(mouse_keyboard_comfort_idx)
        max_idx = max(idxs)
        if len(row) <= max_idx:
            continue
        raw_pid = (row[pid_idx] or "").strip()
        pid = normalize_pid(raw_pid) or raw_pid
        approach = (row[approach_idx] or "").strip()
        fatigue_a = (row[fatigue_a_idx] or "").strip()
        fatigue_c = (row[fatigue_c_idx] or "").strip()
        if familiarity_idx is not None and familiarity_idx < len(row):
            familiarity = (row[familiarity_idx] or "").strip()
        else:
            familiarity = ""
        if mouse_keyboard_comfort_idx is not None and mouse_keyboard_comfort_idx < len(row):
            mouse_keyboard_comfort = (row[mouse_keyboard_comfort_idx] or "").strip()
        else:
            mouse_keyboard_comfort = ""
        # Skip lines that are entirely blank for our fields
        if not (
            pid
            or approach
            or fatigue_a
            or fatigue_c
            or familiarity
            or mouse_keyboard_comfort
        ):
            continue
        records.append({
            "pid": pid,
            "mouse_keyboard_comfort": mouse_keyboard_comfort,
            "familiarity": familiarity,
            "approach": approach,
            "fatigue_a": fatigue_a,
            "fatigue_c": fatigue_c,
        })
    # sort by PID for stable output
    records.sort(key=lambda r: r.get("pid", ""))
    return records


def _write_pickle(payload: Dict[str, object], pickle_path: Path) -> None:
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with pickle_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _read_pickle(pickle_path: Path) -> Dict[str, object]:
    with pickle_path.open("rb") as handle:
        data = pickle.load(handle)
    if isinstance(data, list):
        # Legacy format: just the subset records list.
        return {"subset_records": data, "source_csv": None}
    if not isinstance(data, dict):
        raise TypeError(f"Survey pickle at {pickle_path} did not contain a mapping")
    return data


def load_survey_records(
    *,
    pickle_path: Path | None = None,
    csv_path: Path | None = None,
    refresh: bool = False,
) -> List[Dict[str, str]]:
    payload = load_survey_payload(
        pickle_path=pickle_path,
        csv_path=csv_path,
        refresh=refresh,
    )
    records = payload.get("subset_records")
    if not isinstance(records, list):
        raise TypeError("Survey payload missing 'subset_records' list")
    return records


def load_survey_payload(
    *,
    pickle_path: Path | None = None,
    csv_path: Path | None = None,
    refresh: bool = False,
) -> Dict[str, object]:
    """Load cached survey records, refreshing from the CSV if needed.

    Prefer loading from the pickle for speed/stability. If the pickle is missing
    or stale relative to the CSV, regenerate it so downstream analyses get
    consistent records regardless of the source format.
    """

    target_pickle = pickle_path or DEFAULT_PICKLE_PATH
    csv_candidate = csv_path or DEFAULT_CSV_PATH
    try:
        resolved_csv = _resolve_csv_path(csv_candidate)
    except FileNotFoundError:
        if refresh or not target_pickle.exists():
            raise
        resolved_csv = None

    should_refresh = refresh
    if target_pickle.exists() and not should_refresh and resolved_csv is not None:
        try:
            pickle_mtime = target_pickle.stat().st_mtime
            csv_mtime = resolved_csv.stat().st_mtime
            if csv_mtime > pickle_mtime:
                should_refresh = True
        except OSError:
            should_refresh = True

    if target_pickle.exists() and not should_refresh:
        try:
            payload = _read_pickle(target_pickle)
            records = payload.get("subset_records")
            if isinstance(records, list):
                return payload
            should_refresh = True
        except Exception:
            # Fall back to rebuilding from the CSV if the pickle is unreadable.
            should_refresh = True

    if resolved_csv is None:
        # No CSV available but we were asked to refresh; at this point the
        # caller either forced refresh without providing a CSV or the cached
        # pickle was unreadable. Surface a clear error.
        raise FileNotFoundError(f"Survey CSV not found: {csv_candidate}")

    payload = build_survey_payload(resolved_csv)
    try:
        _write_pickle(payload, target_pickle)
    except Exception:
        # Avoid failing the analysis if the pickle cannot be written; downstream
        # steps can still proceed with the fresh records.
        pass
    return payload


def build_survey_payload(csv_path: Path) -> Dict[str, object]:
    """Build the cached survey payload (subset + demographics records)."""

    subset_records = extract_records(csv_path)
    demographics_records = extract_demographics_records(csv_path)
    return {
        "subset_records": subset_records,
        "demographics_records": demographics_records,
        "source_csv": str(csv_path),
    }


def extract_demographics_records(csv_path: Path) -> List[Dict[str, str]]:
    """Collect rows needed for demographics reporting with canonical PIDs."""

    required_labels = [
        PID_COLUMN_LABEL,
        FINISHED_COLUMN_LABEL,
        SEX_COLUMN_LABEL,
        AGE_COLUMN_LABEL,
        DEGREE_COLUMN_LABEL,
        HISPANIC_COLUMN_LABEL,
        RACE_COLUMN_LABEL,
        ENJOY_C_COLUMN_LABEL,
        ENJOY_A_COLUMN_LABEL,
    ]

    _short, label_row, _meta, data_rows = _read_csv_rows(csv_path)
    column_map: Dict[str, int] = {}
    for label in required_labels:
        try:
            column_map[label] = _find_column(label_row, label)
        except ValueError:
            column_map[label] = -1

    records: List[Dict[str, str]] = []
    pid_idx = column_map.get(PID_COLUMN_LABEL, -1)
    finished_idx = column_map.get(FINISHED_COLUMN_LABEL, -1)

    for row in data_rows:
        if pid_idx < 0 or pid_idx >= len(row):
            continue
        raw_pid = (row[pid_idx] or "").strip()
        canonical = normalize_pid(raw_pid)
        if not canonical:
            continue

        # Honour the Qualtrics Finished flag when present
        if 0 <= finished_idx < len(row):
            finished_text = (row[finished_idx] or "").strip().lower()
            if finished_text and finished_text != "true":
                continue

        record: Dict[str, str] = {
            "pid_canonical": canonical,
            "pid_raw": raw_pid,
        }
        for label, idx in column_map.items():
            if idx < 0 or idx >= len(row):
                record[label] = ""
            else:
                record[label] = (row[idx] or "").strip()
        records.append(record)

    records.sort(key=lambda r: r.get("pid_canonical", ""))
    return records


def load_survey_subset_df(csv_path: Path | None):  # -> pd.DataFrame
    """Return a pandas DataFrame with only the selected fields.

    Requires pandas to be installed; otherwise raises ImportError.
    """
    if pd is None:  # pragma: no cover - optional dep not present
        raise ImportError("pandas is not installed; use extract_records() or install pandas")
    refresh = csv_path is not None and csv_path != DEFAULT_CSV_PATH
    records = load_survey_records(csv_path=csv_path, refresh=refresh)
    df = pd.DataFrame.from_records(
        records,
        columns=[
            "pid",
            "mouse_keyboard_comfort",
            "familiarity",
            "approach",
            "fatigue_a",
            "fatigue_c",
        ],
        dtype=object,
    )
    return df


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract survey subset to CSV table")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="Path to the Qualtrics survey CSV export")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Destination CSV file")
    parser.add_argument(
        "--out-pickle",
        type=Path,
        default=DEFAULT_PICKLE_PATH,
        help="Destination pickle file for cached survey data",
    )
    parser.add_argument(
        "--skip-pickle",
        action="store_true",
        help="Skip writing the survey pickle (CSV only)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    csv_path = _resolve_csv_path(args.csv)
    records = extract_records(csv_path)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Write CSV manually to avoid pandas dependency for CLI
    import csv as _csv
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = _csv.DictWriter(
            handle,
            fieldnames=[
                "pid",
                "mouse_keyboard_comfort",
                "familiarity",
                "approach",
                "fatigue_a",
                "fatigue_c",
            ],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row)
    print(
        f"Wrote {len(records)} rows x {len(writer.fieldnames)} cols to {args.out_csv}"
    )

    if not args.skip_pickle:
        payload = {
            "subset_records": records,
            "demographics_records": extract_demographics_records(csv_path),
            "source_csv": str(csv_path),
        }
        _write_pickle(payload, args.out_pickle)
        print(f"Cached survey payload to {args.out_pickle}")


if __name__ == "__main__":  # pragma: no cover
    main()

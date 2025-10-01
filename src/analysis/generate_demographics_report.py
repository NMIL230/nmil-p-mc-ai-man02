#!/usr/bin/env python3
"""Build a survey demographics report and cross-check AMLEC exclusion status.

The report includes:
- Total survey completions and participant identifiers (PIDs)
- Demographic breakdowns (sex, age descriptive stats, highest degree, ethnicity)
- Enjoyment scores (mean and standard deviation) for each Minecraft mode
- A reconciliation against the gameplay analysis pipeline to list validity and
  outlier exclusions alongside any unmatched participants

Usage (defaults to latest survey export and default AMLEC pickle):
    python -m src.analysis.generate_demographics_report

Optional arguments:
    --survey-csv PATH   Qualtrics export (with the three standard header rows)
    --pickle PATH       Alternate AMLEC dataset pickle
    --out PATH          Write the textual report to a file instead of stdout
"""

from __future__ import annotations

import argparse
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.extract_survey_subset import (
    DEFAULT_CSV_PATH as DEFAULT_SURVEY_CSV,
    DEFAULT_PICKLE_PATH as DEFAULT_SURVEY_PICKLE,
    AGE_COLUMN_LABEL,
    DEGREE_COLUMN_LABEL,
    ENJOY_A_COLUMN_LABEL,
    ENJOY_C_COLUMN_LABEL,
    FINISHED_COLUMN_LABEL,
    HISPANIC_COLUMN_LABEL,
    PID_COLUMN_LABEL,
    RACE_COLUMN_LABEL,
    SEX_COLUMN_LABEL,
    load_survey_payload,
)
from src.utils.data.common_inputs import build_fits_and_pairs
from src.utils.data.pairing import filter_valid_psi_pairs
from src.utils.data.pids import normalize_pid

SURVEY_DIR = DEFAULT_SURVEY_CSV.parent
SURVEY_GLOB = "Active+Machine+Learning+for+Evaluating+Cognition_*.csv"


class ReportError(RuntimeError):
    """Raised when the demographics report cannot be constructed."""


def _find_latest_survey_csv() -> Path:
    candidates = sorted(SURVEY_DIR.glob(SURVEY_GLOB))
    if not candidates:
        raise ReportError(f"No survey CSV found under {SURVEY_DIR}")
    return candidates[-1]


def _load_survey_dataframe(
    csv_path: Optional[Path],
    pickle_path: Optional[Path],
    *,
    refresh: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Return the cached survey demographics DataFrame and payload."""

    payload = load_survey_payload(
        csv_path=csv_path,
        pickle_path=pickle_path,
        refresh=refresh,
    )
    records = payload.get("demographics_records", [])
    if not isinstance(records, list) or not records:
        raise ReportError("Survey demographics payload is empty; regenerate the cache from the CSV.")

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ReportError("Survey demographics payload produced an empty DataFrame")

    df[PID_COLUMN_LABEL] = df[PID_COLUMN_LABEL].astype(str).str.strip()
    df["pid_canonical"] = df["pid_canonical"].astype(str)
    return df.reset_index(drop=True), payload


def _numeric_from_text(series: pd.Series) -> pd.Series:
    """Best-effort numeric coercion for fields that may include labels (e.g., "5 (Liked)")."""
    extracted = series.astype(str).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _describe_series(series: pd.Series) -> Tuple[Optional[float], Optional[float], int]:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.dropna()
    if numeric.empty:
        return None, None, 0
    return float(numeric.mean()), float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0, len(numeric)


def _collect_demographics(
    df: pd.DataFrame,
    restrict_to: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Return demographics restricted to the provided PID set (if any)."""

    if restrict_to is not None:
        pid_set = {str(pid) for pid in restrict_to}
        df = df[df["pid_canonical"].isin(pid_set)].copy()

    result: Dict[str, Any] = {}
    result["pids"] = df["pid_canonical"].dropna().sort_values().tolist()
    result["count"] = len(result["pids"])
    result["sex_counts"] = df[SEX_COLUMN_LABEL].dropna().value_counts().to_dict()
    age_numeric = pd.to_numeric(df[AGE_COLUMN_LABEL], errors="coerce")
    age_numeric = age_numeric.dropna()
    result["age_mean"] = float(age_numeric.mean()) if not age_numeric.empty else None
    result["age_std"] = float(age_numeric.std(ddof=1)) if len(age_numeric) > 1 else None
    result["age_min"] = float(age_numeric.min()) if not age_numeric.empty else None
    result["age_max"] = float(age_numeric.max()) if not age_numeric.empty else None
    result["age_count"] = int(age_numeric.size)
    result["degree_counts"] = df[DEGREE_COLUMN_LABEL].dropna().value_counts().to_dict()
    result["hispanic_counts"] = df[HISPANIC_COLUMN_LABEL].dropna().value_counts().to_dict()
    result["race_counts"] = df[RACE_COLUMN_LABEL].dropna().value_counts().to_dict()
    enjoy_c = _numeric_from_text(df[ENJOY_C_COLUMN_LABEL])
    enjoy_a = _numeric_from_text(df[ENJOY_A_COLUMN_LABEL])
    result["enjoy_c_stats"] = _describe_series(enjoy_c)
    result["enjoy_a_stats"] = _describe_series(enjoy_a)
    result["enjoy_counts"] = {
        "mode_c": enjoy_c.dropna().value_counts().sort_index().to_dict(),
        "mode_a": enjoy_a.dropna().value_counts().sort_index().to_dict(),
    }
    result["demographic_table"] = pd.DataFrame(
        {
            "pid": df["pid_canonical"],
            "sex": df[SEX_COLUMN_LABEL],
            "age": pd.to_numeric(df[AGE_COLUMN_LABEL], errors="coerce"),
            "degree": df[DEGREE_COLUMN_LABEL],
            "hispanic": df[HISPANIC_COLUMN_LABEL],
            "race": df[RACE_COLUMN_LABEL],
        }
    ).sort_values("pid")
    return result


def _collect_exclusion_info(pickle_path: Optional[Path]) -> Dict[str, Any]:
    _, _, ent_pairs, _pairs, valid_ent_pairs, _labels, _gp, _pts = build_fits_and_pairs(pickle_path)
    _, _, _, validity_report = filter_valid_psi_pairs(ent_pairs, min_allowed=0.0, max_allowed=18.0)
    validity_removed: Dict[str, Dict[str, Any]] = {}
    for entry in validity_report.get("skipped_examples", []) if isinstance(validity_report, dict) else []:
        pid = normalize_pid(entry.get("participant")) or str(entry.get("participant"))
        reasons = [str(r) for r in entry.get("reasons", [])]
        validity_removed[pid] = {
            "classic": entry.get("classic"),
            "adaptive": entry.get("adaptive"),
            "reasons": reasons,
        }
    valid_pairs = list(valid_ent_pairs)
    diffs = {pid: float(adaptive) - float(classic) for pid, classic, adaptive in valid_pairs}
    diffs_array = np.array(list(diffs.values()), dtype=float)
    lo = hi = None
    outlier_removed: Dict[str, Dict[str, Any]] = {}
    if diffs_array.size >= 3 and np.isfinite(diffs_array).all():
        q1, q3 = np.percentile(diffs_array, [25, 75])
        iqr = q3 - q1
        lo = float(q1 - 1.5 * iqr)
        hi = float(q3 + 1.5 * iqr)
        for pid, diff in diffs.items():
            if diff < lo or diff > hi:
                outlier_removed[pid] = {"diff": diff, "bounds": (lo, hi)}
    included_after_filters = sorted(set(diffs.keys()) - set(outlier_removed.keys()))
    ent_all = sorted({normalize_pid(pid) or pid for pid, _x, _y in ent_pairs})
    return {
        "ent_pairs": ent_pairs,
        "valid_pairs": valid_pairs,
        "validity_removed": validity_removed,
        "outlier_removed": outlier_removed,
        "outlier_bounds": (lo, hi) if lo is not None and hi is not None else None,
        "included": included_after_filters,
        "all_perf_pids": ent_all,
    }


def _summarize_alignment(
    survey_pids: Iterable[str],
    exclusion_info: Dict[str, Any],
) -> Tuple[pd.DataFrame, List[str]]:
    survey_set = {normalize_pid(pid) or pid for pid in survey_pids}
    all_perf = set(exclusion_info.get("all_perf_pids", []))
    validity_removed = exclusion_info.get("validity_removed", {})
    outlier_removed = exclusion_info.get("outlier_removed", {})
    included = set(exclusion_info.get("included", []))

    rows: List[Dict[str, Any]] = []
    notes: List[str] = []
    for pid in sorted(survey_set):
        record: Dict[str, Any] = {"pid": pid, "status": "included", "details": ""}
        if pid not in all_perf:
            record["status"] = "missing_performance"
            record["details"] = "No $\psi_{\theta}$ pair available"
        elif pid in validity_removed:
            record["status"] = "validity_removed"
            reasons = "; ".join(validity_removed[pid].get("reasons", [])) or "Reason unavailable"
            record["details"] = reasons
        elif pid in outlier_removed:
            record["status"] = "outlier_removed"
            bounds = outlier_removed[pid].get("bounds")
            diff = outlier_removed[pid].get("diff")
            if bounds is not None and diff is not None:
                record["details"] = (
                    f"Diff={diff:.3f} outside [{bounds[0]:.3f}, {bounds[1]:.3f}]"
                )
            else:
                record["details"] = "Flagged as outlier"
        elif pid not in included:
            record["status"] = "excluded_unknown"
            record["details"] = "Missing from final set without recorded reason"
            notes.append(f"PID {pid} missing without exclusion reason")
        rows.append(record)

    # Identify performance PIDs absent from the survey (useful for mismatches)
    extra_perf = sorted(all_perf - survey_set)
    if extra_perf:
        notes.append(
            "Performance dataset includes PIDs not found in survey: " + ", ".join(extra_perf)
        )

    return pd.DataFrame(rows), notes


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _build_demographic_table(
    survey_df: pd.DataFrame,
    table_pids: Sequence[str],
    included: Iterable[str],
    validity_removed: Mapping[str, Any],
    outlier_removed: Mapping[str, Any],
    special_reasons: Mapping[str, str],
) -> pd.DataFrame:
    included_set = {str(pid) for pid in included}
    validity_set = {str(pid) for pid in validity_removed}
    outlier_set = {str(pid) for pid in outlier_removed}
    survey_lookup = (
        survey_df.set_index("pid_canonical").to_dict(orient="index")
        if not survey_df.empty
        else {}
    )

    rows: List[Dict[str, Any]] = []
    for pid in sorted({str(pid) for pid in table_pids}):
        row: Dict[str, Any] = {"pid": pid}
        survey_row = survey_lookup.get(pid, {})
        is_included = pid in included_set
        row.update(
            {
                "sex": _format_cell(survey_row.get(SEX_COLUMN_LABEL)),
                "age": _format_cell(survey_row.get(AGE_COLUMN_LABEL)),
                "degree": _format_cell(survey_row.get(DEGREE_COLUMN_LABEL)),
                "hispanic": _format_cell(survey_row.get(HISPANIC_COLUMN_LABEL)),
                "race": _format_cell(survey_row.get(RACE_COLUMN_LABEL)),
            }
        )

        reason = ""
        if not is_included:
            if pid in validity_set:
                reason = "validity"
            elif pid in outlier_set:
                reason = "outlier"
            else:
                reason = special_reasons.get(pid, "")
        row["excluded?"] = "no" if is_included else "yes"
        row["reason"] = reason
        rows.append(row)

    columns = [
        "pid",
        "sex",
        "age",
        "degree",
        "hispanic",
        "race",
        "excluded?",
        "reason",
    ]
    return pd.DataFrame(rows, columns=columns)


def _format_counts(counts: Dict[str, int]) -> str:
    if not counts:
        return "(none)"
    parts = [f"{key}: {value}" for key, value in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return ", ".join(parts)


def build_report(
    survey_df: pd.DataFrame,
    demographics: Dict[str, Any],
    alignment_df: pd.DataFrame,
    alignment_notes: Sequence[str],
) -> str:
    buf = StringIO()
    csv_name = survey_df.attrs.get("source_csv", "(unknown CSV)")
    buf.write(f"Survey source: {csv_name}\n")
    buf.write(f"Total completed surveys: {demographics['count']}\n")
    buf.write("Participant PIDs: " + ", ".join(demographics["pids"]) + "\n\n")

    buf.write("Demographics\n")
    buf.write("-----------\n")
    buf.write("Sex distribution: " + _format_counts(demographics["sex_counts"]) + "\n")
    if demographics["age_count"]:
        buf.write(
            "Age (n={count}): mean={mean:.1f}, std={std:.1f}, min={min_val:.0f}, max={max_val:.0f}\n".format(
                count=demographics["age_count"],
                mean=demographics["age_mean"],
                std=demographics["age_std"] if demographics["age_std"] is not None else 0.0,
                min_val=demographics["age_min"],
                max_val=demographics["age_max"],
            )
        )
    else:
        buf.write("Age responses unavailable\n")
    buf.write("Highest degree: " + _format_counts(demographics["degree_counts"]) + "\n")
    buf.write("Hispanic / Latino: " + _format_counts(demographics["hispanic_counts"]) + "\n")
    buf.write("Race selection: " + _format_counts(demographics["race_counts"]) + "\n\n")

    buf.write("Enjoyment Scores\n")
    buf.write("----------------\n")
    c_mean, c_std, c_n = demographics["enjoy_c_stats"]
    a_mean, a_std, a_n = demographics["enjoy_a_stats"]
    if c_n:
        buf.write(f"Mode C (n={c_n}): mean={c_mean:.2f}, std={c_std:.2f}\n")
    else:
        buf.write("Mode C: no numeric responses\n")
    if a_n:
        buf.write(f"Mode A (n={a_n}): mean={a_mean:.2f}, std={a_std:.2f}\n")
    else:
        buf.write("Mode A: no numeric responses\n")
    buf.write("Value counts (Mode C): " + _format_counts(demographics["enjoy_counts"]["mode_c"]) + "\n")
    buf.write("Value counts (Mode A): " + _format_counts(demographics["enjoy_counts"]["mode_a"]) + "\n\n")

    buf.write("Participant Breakdown\n")
    buf.write("---------------------\n")
    report_cols = ["pid", "sex", "age", "degree", "hispanic", "race"]
    buf.write(demographics["demographic_table"][report_cols].to_string(index=False))
    buf.write("\n\n")

    buf.write("Exclusion Alignment\n")
    buf.write("--------------------\n")
    buf.write(alignment_df.to_string(index=False))
    buf.write("\n")
    if alignment_notes:
        buf.write("Notes:\n")
        for line in alignment_notes:
            buf.write(f"- {line}\n")
    return buf.getvalue()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AMLEC survey demographics report")
    parser.add_argument(
        "--survey-csv",
        type=Path,
        default=None,
        help="Qualtrics CSV export (defaults to latest)",
    )
    parser.add_argument(
        "--survey-pickle",
        type=Path,
        default=None,
        help="Cached survey pickle (defaults to data/survey/survey_subset.pkl)",
    )
    parser.add_argument("--pickle", type=Path, default=None, help="Alternate AMLEC dataset pickle")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to save the report text")
    parser.add_argument(
        "--table-out",
        type=Path,
        default=Path("results/demographics/demographics_table.csv"),
        help="CSV path for the demographics table with exclusion annotations",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    refresh = args.survey_csv is not None
    survey_pickle = args.survey_pickle or DEFAULT_SURVEY_PICKLE
    survey_df, payload = _load_survey_dataframe(
        args.survey_csv,
        survey_pickle,
        refresh=refresh,
    )
    source_csv = payload.get("source_csv")
    if not source_csv:
        source_csv = str(args.survey_csv or _find_latest_survey_csv())
    survey_df.attrs["source_csv"] = source_csv

    exclusion_info = _collect_exclusion_info(args.pickle)
    included_set = set(exclusion_info.get("included", []))

    demographics = _collect_demographics(survey_df, restrict_to=included_set)

    survey_pids = survey_df["pid_canonical"].dropna().tolist()
    alignment_df, alignment_notes = _summarize_alignment(survey_pids, exclusion_info)

    special_reasons = {
        "AMLEC_013": "no data due to db failure",
        "AMLEC_017": "language barrier",
        "AMLEC_029": "no data due to db failure",
    }
    table_pids = set(survey_pids) | set(special_reasons.keys())
    table_df = _build_demographic_table(
        survey_df,
        sorted(table_pids),
        included_set,
        exclusion_info.get("validity_removed", {}),
        exclusion_info.get("outlier_removed", {}),
        special_reasons,
    )

    if args.table_out:
        args.table_out.parent.mkdir(parents=True, exist_ok=True)
        table_df.to_csv(args.table_out, index=False)
        print(f"Demographic table written to {args.table_out}")

    report_text = build_report(survey_df, demographics, alignment_df, alignment_notes)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report_text, encoding="utf-8")
        print(f"Report written to {args.out}")
    else:
        print(report_text)


if __name__ == "__main__":
    main()

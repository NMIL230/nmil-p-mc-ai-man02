#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import seaborn as sns
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from scipy import stats as scipy_stats

from src.data.amlec_loader import CH_CLASSIC_LABEL, CH_ADAPTIVE_LABEL
from src.data.amlec_loader import load_amlec_diffmaps
from src.data.pid_order import determine_mode_order_for_pid
from src.utils.modeling.fits import FIT_INIT_GUESS, FIT_CAP_X0, fit_sigmoids_per_entity
from src.utils.plotting.labels import pid_display_label
from src.utils.data.pairing import pair_psis_with_entities, filter_valid_psi_pairs
from src.utils.statistics.outliers import filter_df_by_diff_column
from src.utils.plotting.plotting import (
    CORR_OUTLIER_METHOD_DEFAULT,
    CORR_IQR_MULTIPLIER_DEFAULT,
    CORR_ZSCORE_THRESHOLD_DEFAULT,
    DEFAULT_FIGURE_DPI,
)
from src.utils.reporting.formatting import format_stat
from src.utils.plotting.constants import PSI_THETA_TEX, DELTA_PSI_THETA_TEX
from src.utils.reporting.reporting import format_ci_interval, format_df_value
from src.utils.statistics.bayes import compute_jzs_bf_one_sample
from src.utils.statistics.stats import run_linear_model
import statsmodels.formula.api as smf
from src.utils.data.pids import collapse_repeat_sessions
from src.utils.plotting.figure_naming import figure_output_stem

FIGURE_KEY = "Fig05"
FIGURE_STEM = figure_output_stem(FIGURE_KEY)

MODE_NAME_MAP = {
    "CM": "Classic",
    "AM": "Adaptive",
}


def _mode_readable(code: str) -> str:
    if not code:
        return code
    return MODE_NAME_MAP.get(code.upper(), code)


@dataclass(frozen=True)
class ParityRecord:
    pid: str
    psi_classic: float
    psi_adaptive: float
    first_mode: str

    @property
    def second_mode(self) -> str:
        return "Adaptive" if self.first_mode == "Classic" else "Classic"

    @property
    def psi_first(self) -> float:
        return self.psi_classic if self.first_mode == "Classic" else self.psi_adaptive

    @property
    def psi_second(self) -> float:
        return self.psi_adaptive if self.first_mode == "Classic" else self.psi_classic

    @property
    def psi_diff(self) -> float:
        return self.psi_adaptive - self.psi_classic

    @property
    def delta_second_minus_first(self) -> float:
        return self.psi_second - self.psi_first

    @property
    def parity_label(self) -> str:
        return "CM-AM" if self.first_mode == "Classic" else "AM-CM"


def build_parity_records() -> List[ParityRecord]:
    data_x, data_y, *_rest = load_amlec_diffmaps()
    fits_x = fit_sigmoids_per_entity(
        data_x,
        fit_kwargs=dict(init_guess=FIT_INIT_GUESS, cap_x0=FIT_CAP_X0),
        phantom_failure_at=17.0,
    )
    fits_y = fit_sigmoids_per_entity(data_y, fit_kwargs=dict(init_guess=FIT_INIT_GUESS, cap_x0=FIT_CAP_X0))
    ent_pairs = pair_psis_with_entities(fits_x, fits_y, mode="first_vs_first")
    records: List[ParityRecord] = []
    for pid, psi_classic, psi_adaptive in ent_pairs:
        order = determine_mode_order_for_pid(pid)
        first_mode = "Adaptive" if order == "adaptive_first" else "Classic"
        records.append(
            ParityRecord(pid=str(pid), psi_classic=float(psi_classic), psi_adaptive=float(psi_adaptive), first_mode=first_mode)
        )
    records.sort(key=lambda rec: rec.pid)
    return records


def parity_dataframe(records: Iterable[ParityRecord]) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "pid": rec.pid,
                "pid_label": pid_display_label(rec.pid),
                "psi_classic": rec.psi_classic,
                "psi_adaptive": rec.psi_adaptive,
                "first_mode": rec.first_mode,
                "second_mode": rec.second_mode,
                "parity_label": rec.parity_label,
                "psi_first": rec.psi_first,
                "psi_second": rec.psi_second,
                "psi_diff": rec.psi_diff,
                "delta_second_minus_first": rec.delta_second_minus_first,
            }
            for rec in records
        ]
    )
    df["first_mode"] = pd.Categorical(df["first_mode"], categories=["Classic", "Adaptive"], ordered=True)
    df["second_mode"] = pd.Categorical(df["second_mode"], categories=["Classic", "Adaptive"], ordered=True)
    df["parity_label"] = pd.Categorical(df["parity_label"], categories=["CM-AM", "AM-CM"], ordered=True)
    return df


def parity_long_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    long_df = df.melt(
        id_vars=["pid", "first_mode", "second_mode"],
        value_vars=["psi_classic", "psi_adaptive"],
        var_name="modality_key",
        value_name="psi",
    )
    modality_map = {"psi_classic": "Classic", "psi_adaptive": "Adaptive"}
    long_df["modality"] = long_df["modality_key"].map(modality_map)
    long_df["modality"] = pd.Categorical(long_df["modality"], categories=["Classic", "Adaptive"], ordered=True)
    long_df["first_mode"] = long_df["first_mode"].astype("category")
    return long_df


def session_long_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Construct a long-form DataFrame with two rows per PID: session 1 and 2.

    Columns: pid, session (1|2), mode_at_session (Classic|Adaptive), psi
    """
    rows = []
    for _, row in df.iterrows():
        try:
            pid = str(row["pid"])
            fm = str(row["first_mode"])  # Classic or Adaptive
            smode = str(row["second_mode"])  # Classic or Adaptive
            p1 = float(row["psi_first"]) if np.isfinite(row["psi_first"]) else np.nan
            p2 = float(row["psi_second"]) if np.isfinite(row["psi_second"]) else np.nan
        except Exception:
            continue
        rows.append({"pid": pid, "session": 1, "mode_at_session": fm, "psi": p1})
        rows.append({"pid": pid, "session": 2, "mode_at_session": smode, "psi": p2})
    long_df = pd.DataFrame(rows)
    long_df["mode_at_session"] = pd.Categorical(long_df["mode_at_session"], categories=["Classic", "Adaptive"], ordered=True)
    long_df["session"] = pd.Categorical(long_df["session"], categories=[1, 2], ordered=True)
    return long_df


def estimate_global_session_effect(df: pd.DataFrame) -> float:
    """Estimate global second-vs-first session effect controlling for mode and PID.

    Uses OLS with participant fixed effects: psi ~ C(session) + C(mode_at_session) + C(pid).
    Returns the coefficient for session==2 (relative to session 1 baseline).
    """
    long_df = session_long_dataframe(df)
    clean = long_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["psi", "mode_at_session", "session", "pid"])
    if clean.empty:
        return 0.0
    try:
        model = smf.ols("psi ~ C(session) + C(mode_at_session) + C(pid)", data=clean).fit()
        beta = float(model.params.get("C(session)[T.2]", 0.0))
        if not np.isfinite(beta):
            beta = 0.0
        return beta
    except Exception:
        return 0.0


def apply_session_effect_adjustment(df: pd.DataFrame, effect: float) -> pd.DataFrame:
    """Subtract global session effect from second-session psi and update modality fields.

    Returns a copy of df with adjusted psi_first/psi_second and corresponding psi_classic/psi_adaptive,
    plus recomputed derived columns.
    """
    if not np.isfinite(effect) or abs(effect) < 1e-12:
        return df.copy()
    adjusted = df.copy()
    adjusted["psi_second"] = adjusted["psi_second"].astype(float) - float(effect)
    # Update modality-specific psis
    def _update_row(row):
        fm = row["first_mode"]
        p1 = float(row["psi_first"]) if np.isfinite(row["psi_first"]) else np.nan
        p2 = float(row["psi_second"]) if np.isfinite(row["psi_second"]) else np.nan
        if fm == "Classic":
            row["psi_classic"], row["psi_adaptive"] = p1, p2
        else:
            row["psi_adaptive"], row["psi_classic"] = p1, p2
        row["psi_diff"] = float(row["psi_adaptive"]) - float(row["psi_classic"]) if np.isfinite(row["psi_adaptive"]) and np.isfinite(row["psi_classic"]) else np.nan
        row["delta_second_minus_first"] = float(p2) - float(p1) if np.isfinite(p2) and np.isfinite(p1) else np.nan
        return row
    adjusted = adjusted.apply(_update_row, axis=1)
    return adjusted


def plot_parity_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    color_map = {"CM-AM": "#1f77b4", "AM-CM": "#ff7f0e"}
    clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["psi_classic", "psi_adaptive"])
    if clean.empty:
        return
    colors = clean["parity_label"].map(color_map)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(clean["psi_classic"], clean["psi_adaptive"], c=colors, s=80, alpha=0.8, edgecolor="white", linewidths=0.8)
    lims = [min(clean["psi_classic"].min(), clean["psi_adaptive"].min()) - 0.5,
            max(clean["psi_classic"].max(), clean["psi_adaptive"].max()) + 0.5]
    ax.plot(lims, lims, linestyle="--", color="black", alpha=0.7)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"Classic {PSI_THETA_TEX}")
    ax.set_ylabel(f"Adaptive {PSI_THETA_TEX}")
    ax.set_title("Parity by First Mode")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="CM-AM", markerfacecolor="#1f77b4", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="AM-CM", markerfacecolor="#ff7f0e", markersize=8),
    ]
    ax.legend(handles=handles, loc="upper left")
    for _, row in clean.iterrows():
        label = row.get("pid_label") or pid_display_label(row["pid"])
        ax.annotate(label, (row["psi_classic"], row["psi_adaptive"]), textcoords="offset points", xytext=(4, -6), fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{FIGURE_STEM}_parity_scatter.{ext}", dpi=DEFAULT_FIGURE_DPI)
    plt.close(fig)


def plot_learning_slopes(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x_positions = {"Classic": 0, "Adaptive": 1}
    color_map = {"CM-AM": "#1f77b4", "AM-CM": "#ff7f0e"}
    for _, row in df.iterrows():
        if not (np.isfinite(row["psi_first"]) and np.isfinite(row["psi_second"])):
            continue
        xs = [x_positions[row["first_mode"]], x_positions[row["second_mode"]]]
        ys = [row["psi_first"], row["psi_second"]]
        color = color_map[row["parity_label"]]
        ax.plot(xs, ys, marker="o", color=color, alpha=0.7, linewidth=1.5)
        label = row.get("pid_label") or pid_display_label(row["pid"])
        ax.text(xs[-1] + 0.02, ys[-1], label, fontsize=7, alpha=0.8)
    ax.set_xticks([0, 1], ["First Session", "Second Session"])
    ax.set_ylabel(f"Session {PSI_THETA_TEX}")
    ax.set_title("Learning Trajectories by Starting Mode")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{FIGURE_STEM}_learning_slopes.{ext}", dpi=DEFAULT_FIGURE_DPI)
    plt.close(fig)


def plot_delta_distributions(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    enforce_validity: bool = True,
    validity_min: float = 0.0,
    validity_max: float = 18.0,
    exclude_outliers: bool = True,
    outlier_method: str = CORR_OUTLIER_METHOD_DEFAULT,
    iqr_multiplier: float = CORR_IQR_MULTIPLIER_DEFAULT,
    z_thresh: float = CORR_ZSCORE_THRESHOLD_DEFAULT,
    collapse_repeats: bool = False,
    stats_formats: Optional[List[str]] = None,
    stats_order: Optional[List[str]] = None,
) -> None:
    working = df.copy()
    working = working.replace([np.inf, -np.inf], np.nan)

    if enforce_validity and not working.empty:
        ent_pairs = [
            (str(row["pid"]), row["psi_classic"], row["psi_adaptive"])
            for _, row in working.iterrows()
        ]
        _pairs, _labels, valid_ent_pairs, _report = filter_valid_psi_pairs(
            ent_pairs,
            min_allowed=float(validity_min),
            max_allowed=float(validity_max),
        )
        valid_pids = {pid for (pid, _x, _y) in valid_ent_pairs}
        before_validity = len(working)
        if valid_pids:
            working = working[working["pid"].isin(valid_pids)].copy()
        else:
            working = working.iloc[0:0].copy()
        removed_validity = before_validity - len(working)
        if removed_validity > 0:
            min_bound = format_stat(validity_min, mode="decimal")
            max_bound = format_stat(validity_max, mode="decimal")
            print(
                f"{FIGURE_STEM} validity filter: removed "
                f"{removed_validity}/{before_validity} outside [{min_bound}, {max_bound}]."
            )

    if exclude_outliers and len(working) >= 3:
        filtered, info = filter_df_by_diff_column(
            working,
            diff_column="delta_second_minus_first",
            method=outlier_method,
            iqr_multiplier=float(iqr_multiplier),
            z_thresh=float(z_thresh),
        )
        removed_outliers = info.get("removed", 0)
        if removed_outliers > 0:
            bounds = info.get("bounds")
            bounds_txt = "bounds=None"
            if isinstance(bounds, tuple) and len(bounds) == 2 and all(np.isfinite(b) for b in bounds):
                lo_fmt = format_stat(bounds[0], mode="decimal")
                hi_fmt = format_stat(bounds[1], mode="decimal")
                bounds_txt = f"bounds={lo_fmt}..{hi_fmt}"
            print(
                f"{FIGURE_STEM} outlier filter: removed "
                f"{removed_outliers}/{info.get('before', len(working))} "
                f"({info.get('method', outlier_method)}; {bounds_txt})."
            )
        working = filtered

    if collapse_repeats and not working.empty:
        pid_values = working["pid"].astype(str).tolist()
        collapsed_pids = collapse_repeat_sessions(pid_values)
        selected_positions: List[int] = []
        used_positions: set[int] = set()
        for pid in collapsed_pids:
            for pos, value in enumerate(pid_values):
                if pos in used_positions:
                    continue
                if value == pid:
                    selected_positions.append(pos)
                    used_positions.add(pos)
                    break
        selected_positions.sort()
        removed_repeats = len(pid_values) - len(selected_positions)
        if removed_repeats > 0:
            print(
                f"{FIGURE_STEM} repeat collapse: removed "
                f"{removed_repeats} session(s) in favour of non-repeat data."
            )
        working = working.iloc[selected_positions].copy()

    color_map = {"CM-AM": "#1f77b4", "AM-CM": "#ff7f0e"}
    summaries: List[Dict[str, object]] = []

    clean = working.replace([np.inf, -np.inf], np.nan)

    for label in ["CM-AM", "AM-CM"]:
        vals = clean.loc[clean["parity_label"] == label, "delta_second_minus_first"].dropna().values
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        box_stats = boxplot_stats(vals, whis=1.5)[0]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")

        if vals.size >= 2:
            test_result = scipy_stats.ttest_1samp(vals, popmean=0.0, nan_policy="omit")
            if hasattr(test_result, "statistic"):
                t_stat = float(test_result.statistic)
                p_value = float(test_result.pvalue)
            else:
                t_stat = float(test_result[0])
                p_value = float(test_result[1]) if len(test_result) > 1 else float("nan")
            df_val = float(vals.size - 1)
            if np.isfinite(std_val):
                se = std_val / np.sqrt(vals.size)
                t_crit = scipy_stats.t.ppf(0.975, vals.size - 1)
                ci_low = mean_val - t_crit * se
                ci_high = mean_val + t_crit * se
            else:
                ci_low = ci_high = float("nan")
        else:
            t_stat = float("nan")
            p_value = float("nan")
            df_val = float("nan")
            ci_low = ci_high = float("nan")

        if not np.isfinite(t_stat):
            t_stat = float("nan")
        if not np.isfinite(p_value):
            p_value = float("nan")

        bf10_method = "JZS"
        try:
            bf10 = float(compute_jzs_bf_one_sample(vals))
        except Exception:
            bf10_method = "BIC"
            n_obs = float(vals.size)
            if n_obs >= 2 and np.isfinite(mean_val):
                sse_alt = float(np.sum((vals - mean_val) ** 2))
                sse_null = float(np.sum((vals - 0.0) ** 2))
                eps = 1e-12
                sse_alt = max(sse_alt, eps)
                sse_null = max(sse_null, eps)
                bic_null = n_obs * np.log(sse_null / n_obs) + 1 * np.log(n_obs)
                bic_alt = n_obs * np.log(sse_alt / n_obs) + 2 * np.log(n_obs)
                bf10 = float(np.exp((bic_null - bic_alt) / 2.0))
            else:
                bf10 = float("nan")

        summary = {
            "label": label,
            "values": vals,
            "n": int(vals.size),
            "mean": mean_val,
            "std": std_val,
            "median": float(box_stats["med"]),
            "q1": float(box_stats["q1"]),
            "q3": float(box_stats["q3"]),
            "whislo": float(box_stats["whislo"]),
            "whishi": float(box_stats["whishi"]),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "df": float(df_val),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "test_name": "One-sample t-test vs 0 (paired deltas)",
            "bf10": float(bf10),
            "bf10_method": bf10_method,
        }
        summary["median_sigfig"] = format_stat(summary["median"], mode="decimal")
        summary["mean_sigfig"] = format_stat(summary["mean"], mode="decimal")
        summary["q1_sigfig"] = format_stat(summary["q1"], mode="decimal")
        summary["q3_sigfig"] = format_stat(summary["q3"], mode="decimal")
        summary["whislo_sigfig"] = format_stat(summary["whislo"], mode="decimal")
        summary["whishi_sigfig"] = format_stat(summary["whishi"], mode="decimal")
        summary["std_sigfig"] = format_stat(summary["std"], mode="decimal")
        summary["t_stat_sigfig"] = format_stat(summary["t_stat"], mode="decimal")
        df_text = "NA" if not np.isfinite(summary["df"]) else str(int(summary["df"]))
        summary["ci_text"] = format_ci_interval(summary["ci_low"], summary["ci_high"])
        summary["df_text"] = df_text
        summary["p_fmt"] = f"p={format_stat(summary['p_value'], mode='scientific')}"
        summary["bf_fmt"] = (
            f"BF10={format_stat(summary['bf10'], mode='scientific')}"
            if np.isfinite(summary["bf10"])
            else "BF10=NA"
        )
        summary["bf10_value"] = (
            format_stat(summary["bf10"], mode="scientific") if np.isfinite(summary["bf10"]) else "NA"
        )
        summaries.append(summary)

    if stats_order:
        order_index = {label: idx for idx, label in enumerate(stats_order)}
        def _sort_key(entry):
            return order_index.get(entry.get("label"), len(order_index)), entry.get("label")
        summaries.sort(key=_sort_key)

    if not summaries:
        print(f"{FIGURE_STEM}: no finite data available for delta distributions after filtering.")
        return

    annotations: Dict[str, List[str]] = {}
    for entry in summaries:
        p_value_str = format_stat(entry["p_value"], mode="scientific") if np.isfinite(entry["p_value"]) else "NA"
        if stats_formats:
            mapping = {
                "label": entry["label"],
                "n": entry["n"],
                "delta": entry["mean_sigfig"],
                "mean": entry["mean_sigfig"],
                "ci": entry["ci_text"],
                "ci_text": entry["ci_text"],
                "p": p_value_str,
                "p_fmt": entry["p_fmt"],
                "bf10": entry["bf_fmt"],
                "bf10_value": entry["bf10_value"],
                "bf10_method": entry["bf10_method"],
                "test_name": entry["test_name"],
                "t_stat": entry["t_stat_sigfig"],
                "std": entry["std_sigfig"],
            }
            annotation_lines = [str(fmt).format(**mapping) for fmt in stats_formats]
        else:
            if np.isfinite(entry["bf10"]):
                bf10_str = format_stat(entry["bf10"], mode="scientific")
            else:
                bf10_str = "NA"
            delta_line = f"Δ={entry['mean_sigfig']} ({entry['ci_text']})"
            p_line = f"p={p_value_str}; BF10={bf10_str}"
            annotation_lines = [
                f"n={entry['n']}",
                delta_line,
                p_line,
            ]
        annotations[entry["label"]] = annotation_lines

    group_labels = [entry["label"] for entry in summaries]

    violin_df = working.loc[:, ["parity_label", "delta_second_minus_first"]].copy()
    violin_df = violin_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta_second_minus_first"])
    violin_df = violin_df[violin_df["parity_label"].isin(group_labels)]

    if violin_df.empty:
        print(f"{FIGURE_STEM}: no finite data available for violin plot after filtering.")
        return

    display_labels = group_labels
    violin_df["pretty_label"] = violin_df["parity_label"].astype("category")
    violin_df["pretty_label"] = pd.Categorical(violin_df["pretty_label"], categories=display_labels, ordered=True)

    sns.set(style="white")
    fig_width = max(1, len(display_labels)) * 3.2
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    pastel_map = {
        "CM-AM": "#9bc4eb",
        "AM-CM": "#f9c99a",
    }
    palette_map = {label: pastel_map.get(label, color_map.get(label, "#9bc4eb")) for label in group_labels}

    sns.violinplot(
        data=violin_df,
        x="pretty_label",
        y="delta_second_minus_first",
        order=display_labels,
        hue="pretty_label",
        dodge=False,
        palette=palette_map,
        cut=0,
        inner=None,
        linewidth=0,
        saturation=1.0,
        legend=False,
        ax=ax,
    )

    ax.grid(False)
    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Session Order")
    ax.set_ylabel(f"{DELTA_PSI_THETA_TEX} (Second − First)")
    ax.set_title("Task Mode Order Effect")

    y_min = float(np.nanmin(violin_df["delta_second_minus_first"]))
    y_max = float(np.nanmax(violin_df["delta_second_minus_first"]))
    y_range = y_max - y_min
    if not np.isfinite(y_range) or y_range <= 0:
        y_range = max(abs(y_max), 1.0)
    padding = 0.12 * y_range

    series_by_label: Dict[str, np.ndarray] = {}
    for label in display_labels:
        series = violin_df.loc[violin_df["pretty_label"] == label, "delta_second_minus_first"].to_numpy(dtype=float)
        series = series[np.isfinite(series)]
        series_by_label[label] = series

    lower_extents: List[float] = []
    upper_extents: List[float] = []
    for idx, label in enumerate(display_labels):
        summary = next((s for s in summaries if s["label"] == label), None)
        if summary and np.isfinite(summary["mean"]) and np.isfinite(summary["std"]):
            mean_val = float(summary["mean"])
            std_val = float(summary["std"])
            ax.vlines(idx, mean_val - std_val, mean_val + std_val, color="#4d4d4d", linewidth=2.2, zorder=5)
            ax.plot(idx, mean_val, marker="o", color="#1a1a1a", markersize=6.0, markerfacecolor="white", zorder=6)

        lines = annotations.get(label)
        if not lines:
            continue
        stats_text = "\n".join(lines)
        series = series_by_label.get(label)
        if series is not None and series.size:
            q_low = float(np.nanpercentile(series, 5))
            q_high = float(np.nanpercentile(series, 95))
            group_min = float(np.nanmin(series))
            group_max = float(np.nanmax(series))
            local_range = q_high - q_low
            if not np.isfinite(local_range) or local_range <= 0:
                local_range = max(abs(q_high), 1.0)
        else:
            group_min, group_max = y_min, y_max
            q_low, q_high = group_min, group_max
            local_range = y_range if np.isfinite(y_range) and y_range > 0 else max(abs(y_max), 1.0)

        if idx == 0:
            anchor = q_low if np.isfinite(q_low) else group_min
            offset_low = max(0.08 * local_range, 0.2)
            text_y = anchor - offset_low
            lower_extents.append(text_y)
            va = "top"
        else:
            anchor = q_high if np.isfinite(q_high) else group_max
            offset_high = max(0.15 * local_range, 0.35)
            text_y = anchor + offset_high
            upper_extents.append(text_y)
            va = "bottom"
        ax.text(
            idx,
            text_y,
            stats_text,
            ha="center",
            va=va,
            fontsize=9,
            color="#303030",
            linespacing=1.4,
        )

    xtick_positions = list(range(len(display_labels)))
    xtick_text = [f"{label}\n(n={summaries[i]['n']})" for i, label in enumerate(display_labels)]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_text)
    lower_limit = min([y_min - padding * 0.2] + lower_extents) if lower_extents else y_min - padding * 0.6
    upper_limit = max([y_max + padding * 0.2] + upper_extents) if upper_extents else y_max + padding * 0.6
    ax.set_ylim(lower_limit, upper_limit)
    sns.despine(ax=ax)
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.96))

    print(f"\n{FIGURE_STEM} {DELTA_PSI_THETA_TEX}(Second − First Mode {PSI_THETA_TEX}) summary:")
    for entry in summaries:
        p_value_str = format_stat(entry["p_value"], mode="scientific") if np.isfinite(entry["p_value"]) else "NA"
        mapping = {
            "label": entry["label"],
            "n": entry["n"],
            "delta": entry["mean_sigfig"],
            "mean": entry["mean_sigfig"],
            "median": entry["median_sigfig"],
            "q1": entry["q1_sigfig"],
            "q3": entry["q3_sigfig"],
            "whislo": entry["whislo_sigfig"],
            "whishi": entry["whishi_sigfig"],
            "std": entry["std_sigfig"],
            "ci": entry["ci_text"],
            "ci_text": entry["ci_text"],
            "p": p_value_str,
            "p_fmt": entry["p_fmt"],
            "bf10": entry["bf_fmt"],
            "bf10_value": entry["bf10_value"],
            "bf10_method": entry["bf10_method"],
            "test_name": entry["test_name"],
            "t_stat": entry["t_stat_sigfig"],
        }
        if stats_formats:
            for text in (str(fmt).format(**mapping) for fmt in stats_formats):
                print(f"  {text}")
        else:
            std_text = "std=NA" if not np.isfinite(entry["std"]) else f"std={entry['std_sigfig']}"
            effect_text = f"Effect size (mean Δ)={entry['mean_sigfig']}; {entry['ci_text']}"
            t_text = (
                f"{entry['test_name']}: t({entry['df_text']})={entry['t_stat_sigfig']}, {entry['p_fmt']}"
                if np.isfinite(entry["t_stat"]) and np.isfinite(entry["p_value"])
                else f"{entry['test_name']}: t=NA, p=NA"
            )
            line = (
                f"  n={entry['n']}, "
                f"mean={entry['mean_sigfig']}, "
                f"median={entry['median_sigfig']}, "
                f"q1={entry['q1_sigfig']}, "
                f"q3={entry['q3_sigfig']}, "
                f"whiskers=({entry['whislo_sigfig']}, {entry['whishi_sigfig']}), "
                f"{std_text}, "
                f"{effect_text}, "
                f"{t_text}, "
                f"{entry['bf_fmt']}"
            )
            print(line)

    print("Tests: Paired-sample t-test on Δ (two-tailed); Bayes factor BF10.")

    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{FIGURE_STEM}.{ext}", dpi=DEFAULT_FIGURE_DPI)
    plt.close(fig)

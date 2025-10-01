#!/usr/bin/env python3
"""Run Full Analysis and Generate All Figures.

This orchestration script standardises the execution of all manuscript figures
and analyses. Global toggles defined below control behaviour such as outlier
exclusion, validity masking, and repeat handling. Per-script overrides allow
fine-grained adjustments when specific figures require different settings.

After running every task the script consolidates console outputs into a single
Markdown report so the statistics used in the manuscript are preserved
alongside the commands that produced them.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "manuscript_analysis" / "full_analysis_report.md"
STATS_REPORT_PATH = REPO_ROOT / "manuscript_analysis" / "full_stats_report.md"
PYTHON = os.environ.get("ANALYSIS_PYTHON") or sys.executable or "python"
_DEFAULT_R_HOME = Path("/opt/homebrew/Cellar/r/4.5.1/lib/R")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.plotting.figure_naming import figure_display_name, figure_output_dir, figure_output_stem

# ---------------------------------------------------------------------------
# Global toggles (edit these values to change behaviour across all tasks)
# ---------------------------------------------------------------------------
GLOBAL_OPTIONS: Dict[str, Any] = {
    "exclude_outliers": True,
    "outlier_method": "iqr_diff",
    "iqr_multiplier": 1.5,
    "z_thresh": 3.0,
    "apply_validity_mask": True,
    "exclude_repeats": False,
    "adjust_session_effect": True,
    "show_point_labels": False,
    "show_sort_metric": False,
    "show_classic_threshold": True,
    "use_gp_surface": True,
    "survey_metrics": ["minecraft_familiarity", "mouse_keyboard_comfort"],
    "survey_equivalence_bound": 0.5,
    "survey_pickle": REPO_ROOT / "data/AMLEC_survey.pkl",
    "trial_stats_detail": False,
    "trial_stats_k": None,
    "fig02_seed": None,
}


@dataclass
class ScriptConfig:
    enabled: bool = True
    overrides: Dict[str, Any] = field(default_factory=dict)
    extra_args: List[str] = field(default_factory=list)


SCRIPT_CONFIGS: Dict[str, ScriptConfig] = {
    "Fig02": ScriptConfig(overrides={}),
    "Fig03": ScriptConfig(overrides={}),
    "Fig04": ScriptConfig(overrides={
        "stats_config": {
            "lines": [
                "ICC={icc}",
                "{icc_ci}",
                "R²={r2}",
                "p={pearson_p}",
                "BF10={bf10_value}",
                "n={n}"
            ]
        }
    }),
    "Fig05": ScriptConfig(overrides={
        "adjust_session_effect": False,
    }),
    "Fig07": ScriptConfig(overrides={}),
    "Fig06": ScriptConfig(overrides={}),
    "FigS01": ScriptConfig(enabled=False, overrides={}),
    "FigS02": ScriptConfig(enabled=False, overrides={}),
    "SurveyAnalysis": ScriptConfig(overrides={}),
    "DemographicsSummary": ScriptConfig(overrides={}),
    "TrialStats": ScriptConfig(overrides={}),
}

SUPPLEMENTAL_TASKS = {"FigS01", "FigS02"}


def parse_cli_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full AMLEC analysis suite.")
    tasks = list(globals().get("TASKS", []))
    alias_map: Dict[str, str] = {}
    for task in tasks:
        alias_map[task.name] = task.name
        display_name = figure_display_name(task.name)
        alias_map[display_name] = task.name
    task_names = sorted(alias_map.keys())
    parser.add_argument(
        "--include-supplemental",
        action="store_true",
        help="Include supplementary figures (FigS01 and FigS02) in the run.",
    )
    if tasks:
        parser.add_argument(
            "--only",
            nargs="+",
            choices=task_names,
            metavar="TASK",
            help="Run only the specified tasks (by name).",
        )
        parser.add_argument(
            "--skip",
            nargs="+",
            choices=task_names,
            metavar="TASK",
            help="Skip the specified tasks (by name).",
        )
    else:
        parser.add_argument(
            "--only",
            nargs="+",
            metavar="TASK",
            help="Run only the specified tasks (by name).",
        )
        parser.add_argument(
            "--skip",
            nargs="+",
            metavar="TASK",
            help="Skip the specified tasks (by name).",
        )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List the available task names and exit without running anything.",
    )
    parsed = parser.parse_args(list(argv) if argv is not None else None)
    if parsed.only and parsed.skip:
        parser.error("--only and --skip cannot be used together.")
    if parsed.only:
        parsed.only = [alias_map.get(name, name) for name in parsed.only]
    if parsed.skip:
        parsed.skip = [alias_map.get(name, name) for name in parsed.skip]
    return parsed


# ---------------------------------------------------------------------------
# Runtime configuration helpers
# ---------------------------------------------------------------------------

def resolve_runtime_configs(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, ScriptConfig]]:
    global_options = copy.deepcopy(GLOBAL_OPTIONS)
    script_configs: Dict[str, ScriptConfig] = {
        name: copy.deepcopy(cfg) for name, cfg in SCRIPT_CONFIGS.items()
    }
    if args.include_supplemental:
        for name in SUPPLEMENTAL_TASKS:
            script_configs.setdefault(name, ScriptConfig()).enabled = True
    if args.only:
        requested = set(args.only)
        for task in TASKS:
            cfg = script_configs.setdefault(task.name, ScriptConfig())
            cfg.enabled = task.name in requested
    elif args.skip:
        skipped = set(args.skip)
        for name in skipped:
            script_configs.setdefault(name, ScriptConfig()).enabled = False
    return global_options, script_configs


def list_available_tasks() -> None:
    print("Available tasks:")
    for task in TASKS:
        cfg = SCRIPT_CONFIGS.get(task.name, ScriptConfig())
        status = "enabled" if cfg.enabled else "disabled"
        tags: List[str] = []
        if task.name in SUPPLEMENTAL_TASKS:
            tags.append("supplemental")
        tag_suffix = f" [{', '.join(tags)}]" if tags else ""
        display_name = figure_display_name(task.name)
        alias_suffix = f" ({task.name})" if display_name != task.name else ""
        print(f"  - {display_name}{alias_suffix}{tag_suffix}: {status} — {task.description}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_option(global_opts: Dict[str, Any], overrides: Dict[str, Any], key: str, default: Any = None) -> Any:
    if key in overrides:
        return overrides[key]
    if key in global_opts:
        return global_opts[key]
    return default


@dataclass
class Task:
    name: str
    script: Path
    description: str
    builder: Optional[Callable[[Dict[str, Any], Dict[str, Any]], List[str]]]
    outputs: Iterable[Path] = field(default_factory=list)
    cwd: Path = REPO_ROOT


# ---------------------------------------------------------------------------
# Argument builders per task
# ---------------------------------------------------------------------------

def build_fig02_args(global_opts: Dict[str, Any], overrides: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    seed = get_option(global_opts, overrides, "seed", get_option(global_opts, overrides, "fig02_seed"))
    if seed is not None:
        args.extend(["--seed", str(seed)])
    pairs = get_option(global_opts, overrides, "pairs")
    if pairs:
        args.append("--pairs")
        args.extend([f"{int(p[0])},{int(p[1])}" for p in pairs])
    percentiles = get_option(global_opts, overrides, "percentiles")
    if percentiles:
        args.append("--percentiles")
        args.extend([str(float(p)) for p in percentiles])
    mode = get_option(global_opts, overrides, "mode")
    if mode:
        args.extend(["--mode", str(mode)])
    grid_size = get_option(global_opts, overrides, "grid_size")
    if grid_size:
        args.extend(["--grid-size", str(int(grid_size))])
    random_samples = get_option(global_opts, overrides, "random_samples")
    if random_samples:
        args.extend(["--random-samples", str(int(random_samples))])
    title = get_option(global_opts, overrides, "title")
    if title:
        args.extend(["--title", str(title)])
    output = get_option(global_opts, overrides, "output")
    if output:
        args.extend(["--output", str(output)])
    return args


def build_fig03_args(global_opts: Dict[str, Any], overrides: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    if not get_option(global_opts, overrides, "apply_validity_mask", True):
        args.append("--no-apply-validity-mask")
    else:
        args.append("--apply-validity-mask")
    if get_option(global_opts, overrides, "exclude_outliers", True):
        args.append("--exclude-outliers")
    else:
        args.append("--no-exclude-outliers")
    outlier_method = get_option(global_opts, overrides, "outlier_method")
    if outlier_method:
        args.extend(["--outlier-method", str(outlier_method)])
    iqr_multiplier = get_option(global_opts, overrides, "iqr_multiplier")
    if iqr_multiplier is not None:
        args.extend(["--iqr-multiplier", str(float(iqr_multiplier))])
    z_thresh = get_option(global_opts, overrides, "z_thresh")
    if z_thresh is not None:
        args.extend(["--z-thresh", str(float(z_thresh))])
    sort_mode = get_option(global_opts, overrides, "sort_mode")
    if sort_mode:
        args.extend(["--sort-mode", str(sort_mode)])
    if get_option(global_opts, overrides, "show_sort_metric", False):
        args.append("--show-sort-metric")
    else:
        args.append("--no-show-sort-metric")
    if get_option(global_opts, overrides, "show_classic_threshold", True):
        args.append("--show-classic-threshold")
    else:
        args.append("--no-show-classic-threshold")
    nrows = get_option(global_opts, overrides, "nrows")
    if nrows:
        args.extend(["--nrows", str(int(nrows))])
    ncols = get_option(global_opts, overrides, "ncols")
    if ncols:
        args.extend(["--ncols", str(int(ncols))])
    pickle_path = get_option(global_opts, overrides, "pickle")
    if pickle_path:
        args.extend(["--pickle", str(pickle_path)])
    return args


def build_figs01_args(global_opts: Dict[str, Any], overrides: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    if get_option(global_opts, overrides, "apply_validity_mask", True):
        args.append("--apply-validity-mask")
    else:
        args.append("--no-apply-validity-mask")
    if get_option(global_opts, overrides, "exclude_outliers", True):
        args.append("--exclude-outliers")
    else:
        args.append("--no-exclude-outliers")
    outlier_method = get_option(global_opts, overrides, "outlier_method")
    if outlier_method:
        args.extend(["--outlier-method", str(outlier_method)])
    iqr_multiplier = get_option(global_opts, overrides, "iqr_multiplier")
    if iqr_multiplier is not None:
        args.extend(["--iqr-multiplier", str(float(iqr_multiplier))])
    z_thresh = get_option(global_opts, overrides, "z_thresh")
    if z_thresh is not None:
        args.extend(["--z-thresh", str(float(z_thresh))])
    sort_mode = get_option(global_opts, overrides, "sort_mode")
    if sort_mode:
        args.extend(["--sort-mode", str(sort_mode)])
    if get_option(global_opts, overrides, "show_sort_metric", False):
        args.append("--show-sort-metric")
    else:
        args.append("--no-show-sort-metric")
    if get_option(global_opts, overrides, "show_classic_threshold", True):
        args.append("--show-classic-threshold")
    else:
        args.append("--no-show-classic-threshold")
    if get_option(global_opts, overrides, "use_gp_surface", True):
        args.append("--use-gp-surface")
    else:
        args.append("--no-use-gp-surface")
    nrows = get_option(global_opts, overrides, "nrows")
    if nrows:
        args.extend(["--nrows", str(int(nrows))])
    ncols = get_option(global_opts, overrides, "ncols")
    if ncols:
        args.extend(["--ncols", str(int(ncols))])
    pickle_path = get_option(global_opts, overrides, "pickle")
    if pickle_path:
        args.extend(["--pickle", str(pickle_path)])
    return args


def build_figs02_args(global_opts: Dict[str, Any], overrides: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    if get_option(global_opts, overrides, "apply_validity_mask", True):
        args.append("--apply-validity-mask")
    else:
        args.append("--no-apply-validity-mask")
    if get_option(global_opts, overrides, "exclude_outliers", True):
        args.append("--exclude-outliers")
    else:
        args.append("--no-exclude-outliers")
    outlier_method = get_option(global_opts, overrides, "outlier_method")
    if outlier_method:
        args.extend(["--outlier-method", str(outlier_method)])
    iqr_multiplier = get_option(global_opts, overrides, "iqr_multiplier")
    if iqr_multiplier is not None:
        args.extend(["--iqr-multiplier", str(float(iqr_multiplier))])
    z_thresh = get_option(global_opts, overrides, "z_thresh")
    if z_thresh is not None:
        args.extend(["--z-thresh", str(float(z_thresh))])
    sort_mode = get_option(global_opts, overrides, "sort_mode")
    if sort_mode:
        args.extend(["--sort-mode", str(sort_mode)])
    if get_option(global_opts, overrides, "show_sort_metric", False):
        args.append("--show-sort-metric")
    else:
        args.append("--no-show-sort-metric")
    nrows = get_option(global_opts, overrides, "nrows")
    if nrows:
        args.extend(["--nrows", str(int(nrows))])
    ncols = get_option(global_opts, overrides, "ncols")
    if ncols:
        args.extend(["--ncols", str(int(ncols))])
    pickle_path = get_option(global_opts, overrides, "pickle")
    if pickle_path:
        args.extend(["--pickle", str(pickle_path)])
    return args


def build_fig04_args(global_opts: Dict[str, Any], overrides: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    if get_option(global_opts, overrides, "exclude_outliers", True):
        args.append("--exclude-outliers")
    else:
        args.append("--no-exclude-outliers")
    outlier_method = get_option(global_opts, overrides, "outlier_method")
    if outlier_method:
        args.extend(["--outlier-method", str(outlier_method)])
    iqr_multiplier = get_option(global_opts, overrides, "iqr_multiplier")
    if iqr_multiplier is not None:
        args.extend(["--iqr-multiplier", str(float(iqr_multiplier))])
    z_thresh = get_option(global_opts, overrides, "z_thresh")
    if z_thresh is not None:
        args.extend(["--z-thresh", str(float(z_thresh))])
    if get_option(global_opts, overrides, "adjust_session_effect", True):
        args.append("--adjust-session-effect")
    else:
        args.append("--no-adjust-session-effect")
    if get_option(global_opts, overrides, "exclude_repeats", True):
        args.append("--exclude-repeats")
    if get_option(global_opts, overrides, "show_point_labels", False):
        args.append("--show-labels")
    else:
        args.append("--hide-labels")
    pickle_path = get_option(global_opts, overrides, "pickle")
    if pickle_path:
        args.extend(["--pickle", str(pickle_path)])
    filename_stem = get_option(global_opts, overrides, "filename_stem")
    if filename_stem:
        args.extend(["--filename-stem", str(filename_stem)])
    out_dir = get_option(global_opts, overrides, "out")
    if out_dir:
        args.extend(["--out", str(out_dir)])
    stats_config = get_option(global_opts, overrides, "stats_config")
    if stats_config:
        args.extend(["--stats-config", json.dumps(stats_config)])
    return args


def build_fig05_args(global_opts: Dict[str, Any], overrides: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    if get_option(global_opts, overrides, "exclude_outliers", True):
        args.append("--exclude-outliers")
    else:
        args.append("--no-exclude-outliers")
    outlier_method = get_option(global_opts, overrides, "outlier_method")
    if outlier_method:
        args.extend(["--outlier-method", str(outlier_method)])
    iqr_multiplier = get_option(global_opts, overrides, "iqr_multiplier")
    if iqr_multiplier is not None:
        args.extend(["--iqr-multiplier", str(float(iqr_multiplier))])
    z_thresh = get_option(global_opts, overrides, "z_thresh")
    if z_thresh is not None:
        args.extend(["--z-thresh", str(float(z_thresh))])
    if get_option(global_opts, overrides, "adjust_session_effect", True):
        args.append("--adjust-session-effect")
    if get_option(global_opts, overrides, "exclude_repeats", True):
        args.append("--exclude-repeats")
    out_dir = get_option(global_opts, overrides, "out")
    if out_dir:
        args.extend(["--out", str(out_dir)])
    stats_config = get_option(global_opts, overrides, "stats_config")
    if stats_config:
        args.extend(["--stats-config", json.dumps(stats_config)])
    return args


def build_fig07_args(global_opts: Dict[str, Any], overrides: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    if get_option(global_opts, overrides, "exclude_repeats", True):
        args.append("--exclude-repeats")
    if get_option(global_opts, overrides, "exclude_outliers", True):
        args.append("--reject-outliers")
    else:
        args.append("--keep-outliers")
    if get_option(global_opts, overrides, "all_methods", False):
        args.append("--all-methods")
    input_path = get_option(global_opts, overrides, "input")
    if input_path:
        args.extend(["--input", str(input_path)])
    out_path = get_option(global_opts, overrides, "out")
    if out_path:
        args.extend(["--out", str(out_path)])
    rows = get_option(global_opts, overrides, "rows")
    if rows:
        args.extend(["--rows", str(int(rows))])
    cols = get_option(global_opts, overrides, "cols")
    if cols:
        args.extend(["--cols", str(int(cols))])
    return args


def build_fig06_args(global_opts: Dict[str, Any], overrides: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    if get_option(global_opts, overrides, "exclude_repeats", True):
        args.append("--exclude-repeats")
    if get_option(global_opts, overrides, "exclude_outliers", True):
        args.append("--reject-outliers")
    else:
        args.append("--keep-outliers")
    if get_option(global_opts, overrides, "all_methods", False):
        args.append("--all-methods")
    input_path = get_option(global_opts, overrides, "input")
    if input_path:
        args.extend(["--input", str(input_path)])
    out_path = get_option(global_opts, overrides, "out")
    if out_path:
        args.extend(["--out", str(out_path)])
    return args

def build_survey_args(global_opts: Dict[str, Any], overrides: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    if get_option(global_opts, overrides, "exclude_repeats", True):
        args.append("--04-exclude-repeats")
    if get_option(global_opts, overrides, "exclude_outliers", True):
        args.append("--04-exclude-outliers")
    else:
        args.append("--04-no-exclude-outliers")
    outlier_method = get_option(global_opts, overrides, "outlier_method")
    if outlier_method:
        args.extend(["--04-outlier-method", str(outlier_method)])
    iqr_multiplier = get_option(global_opts, overrides, "iqr_multiplier")
    if iqr_multiplier is not None:
        args.extend(["--04-iqr-multiplier", str(float(iqr_multiplier))])
    z_thresh = get_option(global_opts, overrides, "z_thresh")
    if z_thresh is not None:
        args.extend(["--04-z-thresh", str(float(z_thresh))])
    metrics = get_option(global_opts, overrides, "survey_metrics", [])
    if metrics:
        args.extend(["--survey-metric", *map(str, metrics)])
    equivalence = get_option(global_opts, overrides, "survey_equivalence_bound")
    if equivalence is not None:
        args.extend(["--equivalence-bound", str(float(equivalence))])
    out_dir = get_option(global_opts, overrides, "out_dir")
    if out_dir:
        args.extend(["--out-dir", str(out_dir)])
    pickle_path = get_option(global_opts, overrides, "pickle")
    if pickle_path:
        args.extend(["--pickle", str(pickle_path)])
    csv_path = get_option(global_opts, overrides, "survey_csv")
    if csv_path:
        args.extend(["--survey-csv", str(csv_path)])
    survey_pickle = get_option(global_opts, overrides, "survey_pickle")
    if survey_pickle:
        args.extend(["--survey-pickle", str(survey_pickle)])
    return args


def build_demographics_args(global_opts: Dict[str, Any], overrides: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    survey_csv = get_option(global_opts, overrides, "survey_csv")
    if survey_csv:
        args.extend(["--survey-csv", str(survey_csv)])
    survey_pickle = get_option(global_opts, overrides, "survey_pickle")
    if survey_pickle:
        args.extend(["--survey-pickle", str(survey_pickle)])
    pickle_path = get_option(global_opts, overrides, "pickle")
    if pickle_path:
        args.extend(["--pickle", str(pickle_path)])
    summary_default = REPO_ROOT / "manuscript_analysis" / "demographic_summary.csv"
    summary_out = get_option(global_opts, overrides, "summary_out", summary_default)
    if summary_out:
        args.extend(["--summary-out", str(summary_out)])
    table_out = get_option(global_opts, overrides, "table_out")
    if table_out:
        args.extend(["--table-out", str(table_out)])
    out_path = get_option(global_opts, overrides, "out")
    if out_path:
        args.extend(["--out", str(out_path)])
    return args


def build_trial_stats_args(global_opts: Dict[str, Any], overrides: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    if not get_option(global_opts, overrides, "apply_validity_mask", True):
        args.append("--no-validity-mask")
    if not get_option(global_opts, overrides, "exclude_outliers", True):
        args.append("--no-outlier-filter")
    else:
        args.append("--outlier-filter")
    outlier_method = get_option(global_opts, overrides, "outlier_method")
    if outlier_method:
        args.extend(["--outlier-method", str(outlier_method)])
    iqr_multiplier = get_option(global_opts, overrides, "iqr_multiplier")
    if iqr_multiplier is not None:
        args.extend(["--iqr-multiplier", str(float(iqr_multiplier))])
    z_thresh = get_option(global_opts, overrides, "z_thresh")
    if z_thresh is not None:
        args.extend(["--z-thresh", str(float(z_thresh))])
    k_value = get_option(global_opts, overrides, "trial_stats_k")
    if k_value is not None:
        args.extend(["--k", str(float(k_value))])
    if get_option(global_opts, overrides, "trial_stats_detail", False):
        args.append("--detail")
    pickle_path = get_option(global_opts, overrides, "pickle")
    if pickle_path:
        args.extend(["--pickle", str(pickle_path)])
    return args


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
TASKS: List[Task] = [
    Task(
        name="Fig02",
        script=REPO_ROOT / "src/figures/MakeFigure02.py",
        description="Representative pattern sets (Fig 02).",
        builder=build_fig02_args,
        outputs=[figure_output_dir("Fig02", REPO_ROOT / "manuscript_figures")],
    ),
    Task(
        name="Fig03",
        script=REPO_ROOT / "src/figures/MakeFigure03.py",
        description="Adaptive vs. Classic scatter and posterior (Fig 03).",
        builder=build_fig03_args,
        outputs=[figure_output_dir("Fig03", REPO_ROOT / "manuscript_figures")],
    ),
    Task(
        name="Fig04",
        script=REPO_ROOT / "src/figures/MakeFigure04.py",
        description="Classic vs Adaptive threshold parity (Fig 04).",
        builder=build_fig04_args,
        outputs=[figure_output_dir("Fig04", REPO_ROOT / "manuscript_figures")],
    ),
    Task(
        name="Fig05",
        script=REPO_ROOT / "src/figures/MakeFigure05.py",
        description="Mode order practice gains (Fig 05).",
        builder=build_fig05_args,
        outputs=[figure_output_dir("Fig05", REPO_ROOT / "manuscript_figures")],
    ),
    Task(
        name="Fig06",
        script=REPO_ROOT / "src/figures/MakeFigure06.py",
        description="RMSE evolution per sampling method (Fig 06).",
        builder=build_fig06_args,
        outputs=[figure_output_dir("Fig06", REPO_ROOT / "manuscript_figures")],
    ),
    Task(
        name="Fig07",
        script=REPO_ROOT / "src/figures/MakeFigure07.py",
        description="All-method contour overlays (Fig 07).",
        builder=build_fig07_args,
        outputs=[figure_output_dir("Fig07", REPO_ROOT / "manuscript_figures")],
    ),
    Task(
        name="FigS01",
        script=REPO_ROOT / "src/figures/MakeFigureS_X01.py",
        description="Supplementary adaptive probability surfaces (Fig S01).",
        builder=build_figs01_args,
        outputs=[REPO_ROOT / "manuscript_figures" / "FigS01"],
    ),
    Task(
        name="FigS02",
        script=REPO_ROOT / "src/figures/MakeFigureS_X02.py",
        description="Supplementary sigmoid overlay grid (Fig S02).",
        builder=build_figs02_args,
        outputs=[REPO_ROOT / "manuscript_figures" / "FigS02"],
    ),
    Task(
        name="SurveyAnalysis",
        script=REPO_ROOT / "src/analysis/survey_data_and_performance.py",
        description="Survey correlates vs classic performance analysis.",
        builder=build_survey_args,
        outputs=[REPO_ROOT / "manuscript_analysis" / "Survey Data and Performance"],
    ),
    Task(
        name="DemographicsSummary",
        script=REPO_ROOT / "src/analysis/generate_demographics_report.py",
        description="Survey demographics summary export.",
        builder=build_demographics_args,
        outputs=[REPO_ROOT / "manuscript_analysis"],
    ),
    Task(
        name="TrialStats",
        script=REPO_ROOT / "src/analysis/report_trial_stats.py",
        description="Classic vs Adaptive trial-count summary.",
        builder=build_trial_stats_args,
        outputs=[],
    ),
]


@dataclass
class TaskResult:
    task: Task
    command: List[str]
    returncode: int
    stdout: str
    stderr: str


def run_task(task: Task, config: ScriptConfig, global_opts: Dict[str, Any]) -> TaskResult:
    overrides = config.overrides if config else {}
    args = task.builder(global_opts, overrides) if task.builder else []
    if config and config.extra_args:
        args.extend(config.extra_args)
    command = [PYTHON, str(task.script), *args]
    env = build_environment()
    completed = subprocess.run(
        command,
        cwd=task.cwd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    return TaskResult(
        task=task,
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
    )


def ensure_report_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def serialize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    serialised: Dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, Path):
            serialised[key] = str(value)
        elif isinstance(value, list):
            serialised[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            serialised[key] = value
    return serialised


def serialize_script_configs(configs: Dict[str, ScriptConfig]) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for name, cfg in configs.items():
        data[name] = {
            "enabled": cfg.enabled,
            "overrides": serialize_config(cfg.overrides),
            "extra_args": cfg.extra_args,
        }
    return data


def format_command(command: List[str]) -> str:
    return " ".join(command)


def build_environment() -> Dict[str, str]:
    env = os.environ.copy()
    if "R_HOME" not in env and _DEFAULT_R_HOME.exists():
        env["R_HOME"] = str(_DEFAULT_R_HOME)
    r_home = env.get("R_HOME")
    if r_home:
        lib_dir = Path(r_home) / "lib"
        if sys.platform == "darwin" and lib_dir.exists():
            current = env.get("DYLD_LIBRARY_PATH", "")
            paths = [str(lib_dir)]
            if current:
                paths.append(current)
            env["DYLD_LIBRARY_PATH"] = ":".join(paths)
        elif sys.platform.startswith("linux") and lib_dir.exists():
            current = env.get("LD_LIBRARY_PATH", "")
            paths = [str(lib_dir)]
            if current:
                paths.append(current)
            env["LD_LIBRARY_PATH"] = ":".join(paths)
    return env


def write_report(
    results: List[TaskResult],
    global_options: Dict[str, Any],
    script_configs: Dict[str, ScriptConfig],
) -> None:
    ensure_report_parent(REPORT_PATH)
    lines: List[str] = []
    timestamp = datetime.now().isoformat(timespec="seconds")
    lines.append("# Full Analysis Report")
    lines.append(f"Generated: {timestamp}")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    config_payload = {
        "global_options": serialize_config(global_options),
        "script_configs": serialize_script_configs(script_configs),
    }
    lines.append("```json")
    lines.append(json.dumps(config_payload, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    for result in results:
        task = result.task
        cfg = script_configs.get(task.name)
        if cfg and not cfg.enabled:
            continue
        display_name = figure_display_name(task.name)
        if display_name != task.name:
            lines.append(f"### {display_name} ({task.name})")
        else:
            lines.append(f"### {task.name}")
        lines.append(task.description)
        lines.append("")
        lines.append(f"- Command: `{format_command(result.command)}`")
        lines.append(f"- Exit code: {result.returncode}")
        output_list = list(task.outputs)
        if output_list:
            rendered = ", ".join(str(path) for path in output_list)
            lines.append(f"- Outputs: {rendered}")
        lines.append("")
        if result.stdout:
            lines.append("```text")
            lines.append(result.stdout)
            lines.append("```")
            lines.append("")
        if result.stderr:
            lines.append("```text")
            lines.append(result.stderr)
            lines.append("```")
            lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def extract_fig04_stats(stdout: str) -> List[str]:
    lines = stdout.splitlines()
    stats: List[str] = []
    capture = False
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("ICC=") or stripped.startswith("ICC(2,1)="):
            capture = True
        if not capture:
            continue
        if not stripped:
            continue
        if stripped.startswith("Wrote "):
            break
        stats.append(stripped)
        if stripped.startswith("Tests:"):
            break
    return stats


def extract_fig05_stats(stdout: str) -> List[str]:
    stats: List[str] = []
    capture = False
    prefix = figure_output_stem("Fig05")
    for raw in stdout.splitlines():
        stripped = raw.strip()
        if stripped.startswith(f"{prefix} Δ(") or stripped.startswith(f"{prefix} $\\Delta"):
            capture = True
            continue
        if not capture:
            continue
        if not stripped:
            continue
        if stripped.startswith("Wrote "):
            break
        stats.append(stripped)
        if stripped.startswith("Tests:"):
            break
    return stats


def extract_fig06_stats(stdout: str) -> List[str]:
    stats: List[str] = []
    capture = False
    prefix = figure_output_stem("Fig06")
    for raw in stdout.splitlines():
        stripped = raw.strip()
        if stripped.startswith(f"{prefix} Active"):
            capture = True
        if not capture:
            continue
        if not stripped:
            continue
        stats.append(stripped)
    return stats

def extract_survey_stats(stdout: str) -> List[str]:
    stats: List[str] = []
    for raw in stdout.splitlines():
        stripped = raw.strip()
        if stripped.startswith("Paired-sample Pearson correlation"):
            if stats and stats[-1] != "":
                stats.append("")
            stats.append(stripped)
        elif stripped.startswith("Bayes factor BF10"):
            stats.append(stripped)
        elif stripped.startswith("Tests:"):
            stats.append(stripped)
    return stats


STATS_EXTRACTORS: Dict[str, Callable[[str], List[str]]] = {
    "Fig04": extract_fig04_stats,
    "Fig05": extract_fig05_stats,
    "Fig06": extract_fig06_stats,
    "SurveyAnalysis": extract_survey_stats,
}

def write_stats_report(results: List[TaskResult]) -> None:
    ensure_report_parent(STATS_REPORT_PATH)
    lines: List[str] = []
    timestamp = datetime.now().isoformat(timespec="seconds")
    lines.append("# Statistical Summary")
    lines.append(f"Generated: {timestamp}")
    lines.append("")
    for result in results:
        extractor = STATS_EXTRACTORS.get(result.task.name)
        if extractor is None:
            continue
        display_name = figure_display_name(result.task.name)
        if display_name != result.task.name:
            lines.append(f"## {display_name} ({result.task.name})")
        else:
            lines.append(f"## {result.task.name}")
        if result.returncode != 0:
            lines.append(f"Execution failed (exit code {result.returncode}); see full report.")
            lines.append("")
            continue
        stats = extractor(result.stdout)
        if not stats:
            lines.append("No statistical output captured.")
        else:
            lines.extend(stats)
        lines.append("")
    STATS_REPORT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_cli_args(argv)
    if args.list_tasks:
        list_available_tasks()
        return

    global_options, script_configs = resolve_runtime_configs(args)
    only_selection = set(args.only) if args.only else set()
    results: List[TaskResult] = []
    for task in TASKS:
        config = script_configs.setdefault(task.name, ScriptConfig())
        display_name = figure_display_name(task.name)
        if args.only and task.name not in only_selection:
            print(f"Skipping {display_name} (not in --only selection)")
            continue
        if not config.enabled:
            print(f"Skipping {display_name} (disabled in configuration)")
            continue
        print(f"Running {display_name}...")
        result = run_task(task, config, global_options)
        results.append(result)
        if result.returncode != 0:
            print(f"  WARNING: {display_name} exited with {result.returncode}")
        else:
            print(f"  Completed {display_name}")
    write_report(results, global_options, script_configs)
    write_stats_report(results)
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main(sys.argv[1:])

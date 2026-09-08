#
#   Apache License 2.0
#
#   Copyright (c) 2022, Mattias Aabmets
#
#   The contents of this file are subject to the terms and conditions defined in the License.
#   You may not use, modify, or distribute this file except in compliance with the License.
#
#   SPDX-License-Identifier: Apache-2.0
#
"""Bar chart of deap-er mean run time, with DEAP % change when shared."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn

from .timing import DEFAULT_CHART, DEFAULT_JSON, LIB_ER, PCT_CHANGE_FORMULA

_UNIQUE = "#5C6BC0"
_SLOWER = "#9E9E9E"
_GREEN = LinearSegmentedColormap.from_list(
    "deap_er_green",
    ["#C8E6C9", "#66BB6A", "#2E7D32", "#0D3B12"],
)


def format_time(ms: float) -> str:
    """Format a millisecond mean for a bar label.

    Args:
        ms: Mean milliseconds.

    Returns:
        Compact time text.
    """
    if ms >= 100:
        return f"{ms:.1f} ms"
    if ms >= 1:
        return f"{ms:.2f} ms"
    return f"{ms:.3f} ms"


def format_label(case: dict[str, Any]) -> str:
    """Build the on-bar label: time, plus DEAP % change when shared.

    Args:
        case: One report case.

    Returns:
        Label text.
    """
    text = format_time(float(case["deap_er_ms"]))
    if not case.get("shared"):
        return text
    pct = case.get("pct_change")
    if pct is None:
        return text
    return f"{text} ({pct:+.1f}%)"


def _bar_color(case: dict[str, Any], peak_gain: float) -> str:
    if not case.get("shared"):
        return _UNIQUE
    pct = case.get("pct_change")
    if pct is None or pct >= 0.0:
        return _SLOWER
    gain = min(1.0, abs(pct) / max(peak_gain, 1.0))
    return to_hex(_GREEN(Normalize(vmin=0.0, vmax=1.0)(gain)))


def _caption(report: dict[str, Any]) -> str:
    meta = report.get("meta", {})
    repeat = meta.get("repeat", "?")
    warmup = meta.get("warmup", "?")
    formula = meta.get("pct_change_formula", PCT_CHANGE_FORMULA)
    return (
        f"Each bar is the mean wall time of one aabmets/deap-er run. "
        f"Shared cases append {formula} in parentheses "
        f"(negative = faster than DEAP). Unique cases show time only. "
        f"Default mean of {repeat} runs after {warmup} warmups; "
        f"heavier cases record their own repeat in the JSON."
    )


def write_chart(
    report: dict[str, Any],
    output: Path,
    *,
    log_scale: bool = True,
) -> None:
    """Write a bar chart of deap-er mean run time.

    Args:
        report: Document from ``build_report``.
        output: Destination image path.
        log_scale: If True, use a log x-axis.
    """
    cases = list(report["cases"])
    ranked = sorted(cases, key=lambda row: float(row["deap_er_ms"]), reverse=True)
    labels = [case["name"] for case in ranked]
    times = [float(case["deap_er_ms"]) for case in ranked]
    peak_gain = max(
        (abs(float(case["pct_change"])) for case in ranked if case.get("pct_change", 0) < 0),
        default=1.0,
    )
    colors = [_bar_color(case, peak_gain) for case in ranked]
    meta = report.get("meta", {})
    er_ver = str(meta.get("deap_er_version", "unknown"))
    deap_ver = str(meta.get("deap_version", "unknown"))
    height = max(7.4, 0.28 * len(ranked) + 2.6)

    seaborn.set_theme(style="white", context="notebook")
    fig, axis = plt.subplots(figsize=(12.2, height), layout="constrained")
    seaborn.barplot(x=times, y=labels, color=_UNIQUE, ax=axis)
    for bar, color in zip(axis.patches, colors, strict=True):
        bar.set_facecolor(color)
    axis.set_xlabel(f"Mean time of one run (ms) — {LIB_ER} {er_ver}")
    axis.set_ylabel("")
    axis.set_title(f"Hot-path time: {LIB_ER} {er_ver} (DEAP/deap {deap_ver} when shared)")
    fig.supxlabel(_caption(report), fontsize=9, color="#444444")
    if log_scale:
        axis.set_xscale("log")
        axis.set_xlim(min(times) * 0.7, max(times) * 1.7)
        axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    else:
        axis.set_xlim(0, max(times) * 1.22)
    axis.grid(visible=False)
    axis.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    seaborn.despine(ax=axis, bottom=True)
    axis.bar_label(
        axis.containers[0],
        labels=[format_label(case) for case in ranked],
        padding=3,
        fontsize=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def load_report(path: Path) -> dict[str, Any]:
    """Read and validate a bench JSON.

    Args:
        path: Path written by the bench.

    Returns:
        The parsed report object.

    Raises:
        SystemExit: If the file is missing or not a valid bench report.
    """
    if not path.is_file():
        sys.exit(f"Bench JSON not found: {path}\nRun: uv run python tools/perf_bench")
    try:
        report = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        sys.exit(f"Bench JSON is not valid JSON: {path}\n{exc}")
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        sys.exit(f"Bench JSON has no cases: {path}")
    return report


def plot_from_json(
    input_path: Path | None = None,
    output: Path | None = None,
    *,
    log_scale: bool = True,
) -> Path:
    """Load a bench JSON and write the chart.

    Args:
        input_path: Bench JSON path. Defaults to ``reports/hotpath-bench.json``.
        output: Image path. Defaults to ``reports/hotpath-speedups.png``.
        log_scale: If True, use a log x-axis.

    Returns:
        The written image path.
    """
    report = load_report(input_path or DEFAULT_JSON)
    dest = output or DEFAULT_CHART
    write_chart(report, dest, log_scale=log_scale)
    return dest

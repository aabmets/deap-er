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
"""Plot hot-path speeds from a bench JSON, with DEAP/deap as the unit baseline.

Each bar is aabmets/deap-er as a percentage of DEAP/deap speed
(DEAP/deap = 1). Requires ``reports/hotpath-bench.json`` from
``tools/bench_hotpaths.py``.

    uv run python tools/plot_hotpath_speedups.py
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_hex

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_JSON = _REPO_ROOT / "reports" / "hotpath-bench.json"
_DEFAULT_OUTPUT = _REPO_ROOT / "reports" / "hotpath-speedups.png"
_LIB_ER = "aabmets/deap-er"
_BELOW_BASELINE = "#9E9E9E"
_GREEN = LinearSegmentedColormap.from_list(
    "deap_er_green",
    ["#C8E6C9", "#66BB6A", "#2E7D32", "#0D3B12"],
)


class Case(NamedTuple):
    """One timed comparison from the bench JSON."""

    label: str
    before_ms: float
    after_ms: float
    note: str = ""


def _relative_speed(case: Case) -> float:
    return case.before_ms / case.after_ms


def _bar_colors(speeds: list[float]) -> list[str]:
    peak = max(1.01, max(speeds))
    norm = LogNorm(vmin=1.0, vmax=peak)
    colors: list[str] = []
    for speed in speeds:
        if speed < 1.0:
            colors.append(_BELOW_BASELINE)
            continue
        colors.append(to_hex(_GREEN(norm(speed))))
    return colors


def _case_note(name: str) -> str:
    if "clone_individual" in name:
        return "vs deepcopy"
    return ""


def _load_report(path: Path) -> dict[str, Any]:
    """Read and validate the bench JSON.

    Args:
        path: Path written by ``tools/bench_hotpaths.py``.

    Returns:
        The parsed report object.

    Raises:
        SystemExit: If the file is missing or not a valid bench report.
    """
    if not path.is_file():
        sys.exit(f"Bench JSON not found: {path}\nRun: uv run python tools/bench_hotpaths.py")
    try:
        report = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        sys.exit(f"Bench JSON is not valid JSON: {path}\n{exc}")
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        sys.exit(f"Bench JSON has no cases: {path}")
    return report


def _cases_from_report(report: dict[str, Any]) -> tuple[Case, ...]:
    """Turn report rows into plot cases.

    Args:
        report: Parsed bench JSON.

    Returns:
        One ``Case`` per report row.
    """
    loaded: list[Case] = []
    for row in report["cases"]:
        loaded.append(
            Case(
                label=row["name"],
                before_ms=float(row["deap_ms"]),
                after_ms=float(row["deap_er_ms"]),
                note=_case_note(row["name"]),
            )
        )
    return tuple(loaded)


def _percent_of_deap(case: Case) -> str:
    text = f"{_relative_speed(case) * 100:.0f}%"
    if case.note:
        return f"{text} ({case.note})"
    return text


def _library_versions(report: dict[str, Any]) -> tuple[str, str]:
    """Read DEAP and deap-er versions captured in the bench JSON.

    Args:
        report: Parsed bench JSON.

    Returns:
        ``(deap_version, deap_er_version)``.
    """
    meta = report.get("meta", {})
    return (
        str(meta.get("deap_version", "unknown")),
        str(meta.get("deap_er_version", "unknown")),
    )


def _caption(report: dict[str, Any], deap_ver: str, er_ver: str) -> str:
    meta = report.get("meta", {})
    repeat = meta.get("repeat", "?")
    warmup = meta.get("warmup", "?")
    return (
        f"Bars are aabmets/deap-er {er_ver} as a percentage of DEAP/deap "
        f"{deap_ver} speed. Median of {repeat} runs after {warmup} warmups."
    )


def _print_table(cases: tuple[Case, ...], deap_ver: str, er_ver: str) -> None:
    er_header = f"{_LIB_ER} {er_ver}"
    print(f"Relative speed  DEAP/deap {deap_ver} = 1")
    print(f"{'Case':<38} {er_header:>28}")
    for case in cases:
        print(f"{case.label:<38} {_percent_of_deap(case):>28}")


def plot_speedups(
    cases: tuple[Case, ...],
    output: Path,
    *,
    log_scale: bool,
    caption: str,
    deap_ver: str,
    er_ver: str,
) -> None:
    """Write a bar chart of deap-er speed as a percentage of DEAP/deap.

    Args:
        cases: Recorded timings to plot.
        output: Destination image path (suffix selects the format).
        log_scale: If True, use a log x-axis (the default).
        caption: Footer text under the chart.
        deap_ver: DEAP/deap version from the bench report.
        er_ver: aabmets/deap-er version from the bench report.
    """
    ranked = sorted(cases, key=_relative_speed, reverse=True)
    labels = [case.label for case in ranked]
    speeds = [_relative_speed(case) for case in ranked]
    colors = _bar_colors(speeds)

    seaborn.set_theme(style="white", context="notebook")
    fig, axis = plt.subplots(figsize=(11.5, 7.4), layout="constrained")
    seaborn.barplot(x=speeds, y=labels, color=_BELOW_BASELINE, ax=axis)
    for bar, color in zip(axis.patches, colors, strict=True):
        bar.set_facecolor(color)
    axis.set_xlabel(f"Relative speed (DEAP/deap {deap_ver} = 1)")
    axis.set_ylabel("")
    axis.set_title(f"Hot-path speed: aabmets/deap-er {er_ver} vs DEAP/deap {deap_ver}")
    fig.supxlabel(caption, fontsize=9, color="#444444")
    if log_scale:
        axis.set_xscale("log")
        axis.set_xlim(min(speeds) * 0.75, max(speeds) * 1.55)
        axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    else:
        axis.set_xlim(0, max(speeds) * 1.18)
    axis.grid(visible=False)
    axis.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    seaborn.despine(ax=axis, bottom=True)

    axis.bar_label(
        axis.containers[0],
        labels=[_percent_of_deap(case) for case in ranked],
        padding=3,
        fontsize=8,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, print the table, and write the chart.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=_DEFAULT_JSON,
        help=f"bench JSON path (default: {_DEFAULT_JSON})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"image path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="linear x-axis (default is log)",
    )
    args = parser.parse_args(argv)
    report = _load_report(args.input)
    cases = _cases_from_report(report)
    deap_ver, er_ver = _library_versions(report)
    _print_table(cases, deap_ver, er_ver)
    plot_speedups(
        cases,
        args.output,
        log_scale=not args.linear,
        caption=_caption(report, deap_ver, er_ver),
        deap_ver=deap_ver,
        er_ver=er_ver,
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

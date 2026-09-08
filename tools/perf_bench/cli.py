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
"""CLI entry for the hot-path bench and chart."""

from __future__ import annotations

import argparse
from pathlib import Path

from deap_er import creator as er_creator

from .chart import plot_from_json, write_chart
from .data import drop_creator_types, make_unique_types
from .report import build_report, print_table, write_json
from .shared import run_shared_cases
from .skipped import SKIPPED_FEATURES
from .timing import DEFAULT_CHART, DEFAULT_JSON
from .unique_columnar import run_unique_columnar
from .unique_es import run_unique_es
from .unique_gp import run_unique_gp
from .unique_ops import run_unique_ops
from .unique_qd import run_unique_qd


def run_all_cases() -> list:
    """Run shared DEAP comparisons and unique deap-er feature benches.

    Returns:
        Combined case results in report order.
    """
    cases = run_shared_cases()
    make_unique_types()
    try:
        cases.extend(run_unique_ops())
        cases.extend(run_unique_es())
        cases.extend(run_unique_gp())
        cases.extend(run_unique_columnar())
        cases.extend(run_unique_qd())
    finally:
        drop_creator_types(er_creator)
    return cases


def main(argv: list[str] | None = None) -> int:
    """Time components, write JSON, then write the chart.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Time DEAP-shared and deap-er-only hot paths, then chart them."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_JSON,
        help=f"JSON path (default: {DEFAULT_JSON})",
    )
    parser.add_argument(
        "--chart",
        type=Path,
        default=DEFAULT_CHART,
        help=f"chart path (default: {DEFAULT_CHART})",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="linear chart x-axis (default is log)",
    )
    args = parser.parse_args(argv)
    report = build_report(run_all_cases(), skipped=SKIPPED_FEATURES)
    print_table(report)
    write_json(report, args.output)
    print(f"Wrote {args.output}")
    write_chart(report, args.chart, log_scale=not args.linear)
    print(f"Wrote {args.chart}")
    return 0


def plot_main(argv: list[str] | None = None) -> int:
    """Plot an existing bench JSON.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Plot hot-path times from a bench JSON.")
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_JSON)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_CHART)
    parser.add_argument("--linear", action="store_true")
    args = parser.parse_args(argv)
    path = plot_from_json(args.input, args.output, log_scale=not args.linear)
    print(f"Wrote {path}")
    return 0

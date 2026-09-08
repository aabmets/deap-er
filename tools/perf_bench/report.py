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
"""JSON report schema and table printing for the hot-path bench."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import deap

from .timing import LIB_DEAP, LIB_ER, PCT_CHANGE_FORMULA, REPEAT, WARMUP, package_version


@dataclass(slots=True)
class CaseResult:
    """One timed component, shared with DEAP or deap-er-only."""

    name: str
    description: str
    shared: bool
    deap_er_ms: float
    deap_ms: float | None = None
    feature: str = ""
    notes: str = ""
    repeat: int = REPEAT
    warmup: int = WARMUP

    @property
    def pct_change(self) -> float | None:
        """Percent change of deap-er vs DEAP, or ``None`` when unique."""
        if self.deap_ms is None or self.deap_ms == 0.0:
            return None
        return (self.deap_er_ms - self.deap_ms) / self.deap_ms * 100.0

    def as_dict(self) -> dict[str, Any]:
        """Serialize this case for JSON.

        Returns:
            A JSON-ready mapping.
        """
        row: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "feature": self.feature or None,
            "shared": self.shared,
            "deap_er_ms": self.deap_er_ms,
            "repeat": self.repeat,
            "warmup": self.warmup,
        }
        if self.notes:
            row["notes"] = self.notes
        if self.shared and self.deap_ms is not None:
            pct = self.pct_change
            row["deap_ms"] = self.deap_ms
            row["pct_change"] = pct
            row["relative_speed"] = self.deap_ms / self.deap_er_ms
        return row


def build_report(
    cases: list[CaseResult],
    *,
    skipped: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Assemble the JSON document from timed cases.

    Args:
        cases: Shared and unique results in display order.
        skipped: Inventory features that are not timed.

    Returns:
        JSON-serializable report.
    """
    libraries: dict[str, dict[str, float]] = {LIB_ER: {}, LIB_DEAP: {}}
    relative: dict[str, float] = {}
    for case in cases:
        libraries[LIB_ER][case.name] = case.deap_er_ms
        if case.shared and case.deap_ms is not None:
            libraries[LIB_DEAP][case.name] = case.deap_ms
            relative[case.name] = case.deap_ms / case.deap_er_ms
    return {
        "meta": {
            "deap_version": package_version("deap"),
            "deap_er_version": package_version("deap-er"),
            "repeat": REPEAT,
            "warmup": WARMUP,
            "deap_module": deap.__file__,
            "pct_change_formula": PCT_CHANGE_FORMULA,
            "skipped_features": skipped or [],
        },
        "libraries": libraries,
        "relative_speed": relative,
        "cases": [case.as_dict() for case in cases],
    }


def print_table(report: dict[str, Any]) -> None:
    """Print a comparison table to stdout.

    Args:
        report: Document from ``build_report``.
    """
    print(f"{'Case':<42} {'shared':>6} {LIB_DEAP:>12} {LIB_ER:>14} {'Δ%':>8}")
    for case in report["cases"]:
        shared = "yes" if case["shared"] else "no"
        deap_ms = case.get("deap_ms")
        deap_txt = f"{deap_ms:11.4f}" if deap_ms is not None else f"{'—':>11}"
        pct = case.get("pct_change")
        pct_txt = f"{pct:7.1f}" if pct is not None else f"{'—':>7}"
        print(f"{case['name']:<42} {shared:>6} {deap_txt} {case['deap_er_ms']:13.4f} {pct_txt}")


def write_json(report: dict[str, Any], path: Path) -> None:
    """Write the report as indented JSON.

    Args:
        report: Document from ``build_report``.
        path: Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")

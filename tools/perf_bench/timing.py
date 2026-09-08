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
"""Shared timing helpers and default report paths."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

REPEAT = 50
WARMUP = 2
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_JSON = REPO_ROOT / "reports" / "hotpath-bench.json"
DEFAULT_CHART = REPO_ROOT / "reports" / "hotpath-speedups.png"
LIB_DEAP = "DEAP/deap"
LIB_ER = "aabmets/deap-er"
PCT_CHANGE_FORMULA = "(deap_er_ms - deap_ms) / deap_ms * 100"


def mean_ms(
    func: Callable[[], object],
    *,
    repeat: int = REPEAT,
    warmup: int = WARMUP,
) -> float:
    """Return the arithmetic-mean runtime of ``func`` in milliseconds.

    Args:
        func: Nullary callable to time.
        repeat: Timed repetitions after warmup.
        warmup: Untimed calls to run first.

    Returns:
        Mean elapsed milliseconds.
    """
    for _ in range(warmup):
        func()
    samples: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.fmean(samples)


def package_version(name: str) -> str:
    """Return an installed package version, or ``unknown``.

    Args:
        name: Distribution name on PyPI.

    Returns:
        Version string.
    """
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"

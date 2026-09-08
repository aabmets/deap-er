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
from __future__ import annotations

from typing import Literal

import numpy

__all__: list[str] = [
    "LexicaseMode",
    "apply_epsilon_filter",
    "apply_strict_filter",
    "epsilon_mode_uses_pool_elite",
    "epsilon_mode_uses_pool_mad",
    "slack_for_case",
]

type LexicaseMode = Literal[
    "strict",
    "epsilon_auto",
    "epsilon_static",
    "epsilon_semi",
    "epsilon_dynamic",
    "epsilon_fixed",
]


def epsilon_mode_uses_pool_mad(mode: LexicaseMode) -> bool:
    """Return whether MAD is computed on the current filter pool."""
    return mode in ("epsilon_dynamic",)


def epsilon_mode_uses_pool_elite(mode: LexicaseMode) -> bool:
    """Return whether the elite error is taken from the current filter pool."""
    return mode in ("epsilon_semi", "epsilon_dynamic")


def _mad(vals: numpy.ndarray) -> float:
    median = float(numpy.median(vals))
    return float(numpy.median(numpy.abs(vals - median)))


def slack_for_case(
    col: numpy.ndarray,
    active: numpy.ndarray,
    mode: LexicaseMode,
    epsilon: float | None,
) -> float:
    """Compute epsilon slack for one case column.

    Args:
        col: Case values for every individual.
        active: Current filter-pool mask.
        mode: Epsilon lexicase variant.
        epsilon: Fixed slack when ``mode`` is ``epsilon_fixed``.

    Returns:
        Slack around the elite error for this case.

    Raises:
        ValueError: If ``epsilon`` is missing for ``epsilon_fixed``.
    """
    if mode == "epsilon_fixed":
        if epsilon is None:
            raise ValueError("epsilon must be set for epsilon_fixed mode")
        return float(epsilon)
    vals = col[active] if epsilon_mode_uses_pool_mad(mode) else col
    return _mad(vals)


def apply_strict_filter(
    active: numpy.ndarray,
    col: numpy.ndarray,
    maximize: bool,
) -> numpy.ndarray:
    """Filter candidates to strict elite ties on one case."""
    vals = col[active]
    best = numpy.max(vals) if maximize else numpy.min(vals)
    keep = col == best
    return numpy.where(active, keep, False)


def apply_epsilon_filter(
    active: numpy.ndarray,
    col: numpy.ndarray,
    maximize: bool,
    slack: float,
    *,
    pool_elite: bool,
) -> numpy.ndarray:
    """Filter candidates within ``slack`` of the elite error on one case."""
    vals = col[active] if pool_elite else col
    if maximize:
        bound = numpy.max(vals) - slack
        keep = col >= bound
    else:
        bound = numpy.min(vals) + slack
        keep = col <= bound
    return numpy.where(active, keep, False)

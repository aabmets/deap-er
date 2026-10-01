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
    "rank_non_finite_worst",
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
    return mode in ("epsilon_semi", "epsilon_dynamic", "epsilon_fixed")


def rank_non_finite_worst(
    matrix: numpy.ndarray,
    fit_weights: tuple[float, ...],
) -> numpy.ndarray:
    """Replace every non-finite case value with the worst value for its case.

    NaN and ``±inf`` become ``+inf`` on minimized cases and ``-inf`` on
    maximized ones, so they never pass a case that a finite value passes.

    Args:
        matrix: ``(n_individuals, n_cases)`` case matrix.
        fit_weights: Per-column maximize/minimize signs.

    Returns:
        ``matrix`` itself when every value is finite, otherwise a copy.
    """
    finite = numpy.isfinite(matrix)
    if finite.all():
        return matrix
    worst = numpy.where(numpy.asarray(fit_weights, dtype=float) > 0, -numpy.inf, numpy.inf)
    return numpy.where(finite, matrix, worst)


def _mad(vals: numpy.ndarray) -> float:
    vals = vals[numpy.isfinite(vals)]
    if vals.size == 0:
        return 0.0
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
        Slack around the elite error for this case. The MAD ignores
        non-finite values and is ``0.0`` when none are finite.

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
    """Filter candidates within ``slack`` of the elite error on one case.

    With a population elite (``pool_elite=False``) the case acts as a
    pass/fail test. When no active candidate passes, the case does not
    discriminate and the pool is kept unchanged.
    """
    vals = col[active] if pool_elite else col
    if maximize:
        bound = numpy.max(vals) - slack
        keep = col >= bound
    else:
        bound = numpy.min(vals) + slack
        keep = col <= bound
    survivors = numpy.logical_and(active, keep)
    return survivors if survivors.any() else active

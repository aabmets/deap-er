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

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual
from deap_er.private.various.rng import rng

from .sel_lexicase_matrix import (
    LexicaseMode,
    case_subset,
    fitness_case_matrix,
    lexicase_select_vectorized,
    require_population,
    validate_case_matrix,
)

__all__: list[str] = ["lexicase_select", "sel_lexicase", "sel_epsilon_lexicase"]


def lexicase_select(
    individuals: list[Individual],
    sel_count: int,
    keep: Callable[[list[Individual], int, bool], list[Individual]],
    *,
    cases: Sequence[int] | None = None,
) -> list[Individual]:
    """Select individuals by filtering fitness cases one at a time.

    Cases are considered in a fresh random order for each selection.
    The last remaining candidate wins; ties are broken at random.
    When ``cases`` is given, every draw uses that subset; only the
    filter order is rerolled.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.
        keep: Callable that receives the current candidates, the index
            of the active fitness case, and whether that case is
            maximized, and returns the surviving candidates.
        cases: Fitness-case indices to filter on. All cases are used
            when omitted. The caller's sequence is not mutated.

    Returns:
        The selected individuals.

    Raises:
        IndexError: If the population is empty or a case index is
            outside the fitness length.
    """
    if sel_count <= 0:
        return []
    subset = case_subset(individuals, cases)
    fit_weights = individuals[0].fitness.weights
    selected = []
    for _i in range(sel_count):
        order = list(subset)
        rng.shuffle(order)
        candidates = individuals
        while len(order) > 0 and len(candidates) > 1:
            case = order[0]
            candidates = keep(candidates, case, fit_weights[case] > 0)
            order.pop(0)
        pool = candidates if candidates else individuals
        selected.append(rng.choice(pool))
    return selected


def _resolve_matrix(
    individuals: list[Individual],
    matrix: numpy.ndarray | None,
    *,
    trust_matrix: bool,
) -> numpy.ndarray:
    if matrix is None:
        return fitness_case_matrix(individuals)
    validate_case_matrix(matrix, individuals, trust=trust_matrix)
    return matrix


def sel_lexicase(
    individuals: list[Individual],
    sel_count: int,
    *,
    cases: Sequence[int] | None = None,
    matrix: numpy.ndarray | None = None,
    trust_matrix: bool = False,
) -> list[Individual]:
    """Select individuals by lexicase filtering of fitness cases.

    Each selected individual is the last remaining candidate after
    fitness cases are considered one at a time in random order.
    Pass ``cases`` to restrict the filter to a per-generation subset.
    Pass ``matrix`` when a packed case matrix is already available.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.
        cases: Fitness-case indices to filter on. All cases are used
            when omitted. Rebuild the subset each generation; do not
            freeze it on the toolbox.
        matrix: Optional ``(n_individuals, n_cases)`` case matrix.
            When omitted, values are read from ``fitness.values``.
        trust_matrix: When ``True``, ``matrix`` is accepted on shape
            alone. Defaults to ``False``.

    Returns:
        The selected individuals.

    Raises:
        IndexError: If the population is empty or a case index is
            outside the fitness length.
        ValueError: If ``matrix`` shape or values do not match fitness.
    """
    if sel_count <= 0:
        return []
    require_population(individuals)
    packed = _resolve_matrix(individuals, matrix, trust_matrix=trust_matrix)
    subset = case_subset(individuals, cases)
    return lexicase_select_vectorized(
        individuals,
        sel_count,
        packed,
        subset,
        individuals[0].fitness.weights,
        mode="strict",
    )


def sel_epsilon_lexicase(
    individuals: list[Individual],
    sel_count: int,
    epsilon: float | None = None,
    *,
    mode: LexicaseMode | None = None,
    cases: Sequence[int] | None = None,
    matrix: numpy.ndarray | None = None,
    trust_matrix: bool = False,
) -> list[Individual]:
    """Select individuals by epsilon-lexicase filtering of fitness cases.

    Each selected individual is the last remaining candidate after
    fitness cases are considered one at a time in random order.
    Candidates within ``epsilon`` of the best case value are kept.
    Pass ``cases`` to restrict the filter to a per-generation subset.
    Pass ``matrix`` when a packed case matrix is already available.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.
        epsilon: Slack around the best case value. If omitted, it is
            computed from the median absolute deviation of the case
            values, separately for every case.
        mode: Epsilon variant when ``epsilon`` is omitted:
            ``epsilon_auto`` or ``epsilon_static`` (population MAD and
            elite), ``epsilon_semi`` (population MAD, pool elite), or
            ``epsilon_dynamic`` (pool MAD and elite). Defaults to
            ``epsilon_auto``.
        cases: Fitness-case indices to filter on. All cases are used
            when omitted. Rebuild the subset each generation; do not
            freeze it on the toolbox.
        matrix: Optional ``(n_individuals, n_cases)`` case matrix.
            When omitted, values are read from ``fitness.values``.
        trust_matrix: When ``True``, ``matrix`` is accepted on shape
            alone. Defaults to ``False``.

    Returns:
        The selected individuals.

    Raises:
        IndexError: If the population is empty or a case index is
            outside the fitness length.
        ValueError: If ``matrix`` shape or values do not match fitness.
    """
    if sel_count <= 0:
        return []
    require_population(individuals)
    packed = _resolve_matrix(individuals, matrix, trust_matrix=trust_matrix)
    subset = case_subset(individuals, cases)
    if epsilon is not None:
        resolved = "epsilon_fixed"
    elif mode is None:
        resolved = "epsilon_auto"
    else:
        resolved = mode
    return lexicase_select_vectorized(
        individuals,
        sel_count,
        packed,
        subset,
        individuals[0].fitness.weights,
        mode=resolved,
        epsilon=epsilon,
    )

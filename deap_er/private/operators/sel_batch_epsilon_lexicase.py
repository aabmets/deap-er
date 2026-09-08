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

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

from .case_batch_reduce import CaseReduction, batch_case_matrix, reduce_case_mse
from .sel_lexicase_matrix import (
    case_subset,
    fitness_case_matrix,
    lexicase_select_vectorized,
    require_population,
    validate_case_matrix,
)

__all__: list[str] = ["sel_batch_epsilon_lexicase"]


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


def sel_batch_epsilon_lexicase(
    individuals: list[Individual],
    sel_count: int,
    batch_size: int,
    epsilon: float | None = None,
    *,
    cases: Sequence[int] | None = None,
    matrix: numpy.ndarray | None = None,
    trust_matrix: bool = False,
    reduction: CaseReduction | None = None,
) -> list[Individual]:
    """Select individuals by epsilon-lexicase on batched case reductions.

    Cases are shuffled and grouped into batches of at most
    ``batch_size``. Each batch is reduced to one pseudo-case (mean
    squared error by default), then the usual epsilon-lexicase filter
    runs on the shorter matrix. A fresh shuffle and partition are
    drawn for every selected individual.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.
        batch_size: Maximum cases per batch.
        epsilon: Slack around the best batch value. If omitted, it is
            computed from the median absolute deviation of each batch
            column separately.
        cases: Fitness-case indices to filter on. All cases are used
            when omitted.
        matrix: Optional ``(n_individuals, n_cases)`` case matrix.
            When omitted, values are read from ``fitness.values``.
        trust_matrix: When ``True``, ``matrix`` is accepted on shape
            alone. Defaults to ``False``.
        reduction: Maps a ``(n_individuals, batch_width)`` block to
            one score per individual. Defaults to mean squared error.

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
    fit_weights = individuals[0].fitness.weights
    reduce = reduction if reduction is not None else reduce_case_mse
    mode = "epsilon_auto" if epsilon is None else "epsilon_fixed"
    selected: list[Individual] = []
    for _ in range(sel_count):
        batched, batch_weights = batch_case_matrix(packed, subset, fit_weights, batch_size, reduce)
        selected.extend(
            lexicase_select_vectorized(
                individuals,
                1,
                batched,
                list(range(batched.shape[1])),
                batch_weights,
                mode=mode,
                epsilon=epsilon,
            )
        )
    return selected

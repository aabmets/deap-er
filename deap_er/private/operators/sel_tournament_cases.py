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
from numbers import Integral
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual
from deap_er.private.various.rng import rng

from .case_batch_reduce import CaseReduction, reduce_case_mean
from .sel_lexicase_matrix import (
    case_subset,
    fitness_case_matrix,
    validate_case_matrix,
)

__all__: list[str] = ["sel_tournament_cases"]


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


def _resolve_case_indices(
    individuals: list[Individual],
    cases: Sequence[int] | None,
    case_count: int | None,
) -> list[int]:
    if cases is not None:
        return case_subset(individuals, cases)
    all_cases = case_subset(individuals, None)
    if case_count is None:
        return all_cases
    if isinstance(case_count, bool) or not isinstance(case_count, Integral):
        raise ValueError("case_count must be an int")
    count = int(case_count)
    if count <= 0:
        return []
    if count >= len(all_cases):
        return all_cases
    return rng.sample(all_cases, count)


def _tournament_scores(
    matrix: numpy.ndarray,
    case_indices: list[int],
    fit_weights: tuple[float, ...],
    reduction: CaseReduction,
) -> numpy.ndarray:
    if not case_indices:
        return numpy.zeros(matrix.shape[0], dtype=numpy.float64)
    block = matrix[:, case_indices]
    scores = reduction(block)
    if fit_weights[case_indices[0]] < 0:
        return -scores
    return scores


def sel_tournament_cases(
    individuals: list[Individual],
    rounds: int,
    contestants: int,
    *,
    cases: Sequence[int] | None = None,
    case_count: int | None = None,
    matrix: numpy.ndarray | None = None,
    trust_matrix: bool = False,
    reduction: CaseReduction | None = None,
) -> list[Individual]:
    """Tournament selection on a down-sampled case aggregate.

    Each individual is scored by reducing a case subset to one scalar
    (column mean by default), then standard tournament selection runs
    on those scores. Informed down-sampling stays on
    :func:`~deap_er.tools.sample_informed_cases`; pass its result as
    ``cases=``.

    Args:
        individuals: Individuals to select from.
        rounds: Number of tournament rounds.
        contestants: Number of individuals in each round.
        cases: Fitness-case indices to score. All cases are used when
            omitted and ``case_count`` is not set.
        case_count: When ``cases`` is omitted, draw this many distinct
            case indices at random. Ignored when ``cases`` is given.
        matrix: Optional ``(n_individuals, n_cases)`` case matrix.
            When omitted, values are read from ``fitness.values``.
        trust_matrix: When ``True``, ``matrix`` is accepted on shape
            alone. Defaults to ``False``.
        reduction: Maps a ``(n_individuals, n_cases)`` block to one
            score per individual. Defaults to the column mean.

    Returns:
        The selected individuals.

    Raises:
        IndexError: If the population is empty.
        ValueError: If ``contestants`` is not positive or ``matrix``
            shape or values do not match fitness.
    """
    if rounds <= 0:
        return []
    n = len(individuals)
    if n == 0:
        raise IndexError("Cannot choose from an empty sequence")
    if contestants < 1:
        raise ValueError("contestants must be at least 1")
    packed = _resolve_matrix(individuals, matrix, trust_matrix=trust_matrix)
    case_indices = _resolve_case_indices(individuals, cases, case_count)
    reduce = reduction if reduction is not None else reduce_case_mean
    scores = _tournament_scores(packed, case_indices, individuals[0].fitness.weights, reduce)
    idxs = rng.integers(0, n, size=rounds * contestants)
    if contestants == 1:
        return [individuals[idx] for idx in idxs]
    chosen: list[Individual] = []
    for i in range(0, rounds * contestants, contestants):
        winner = individuals[idxs[i]]
        best = scores[idxs[i]]
        for offset in range(1, contestants):
            candidate_idx = idxs[i + offset]
            score = scores[candidate_idx]
            if score > best:
                winner = individuals[candidate_idx]
                best = score
        chosen.append(winner)
    return chosen

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
from deap_er.private.various.rng import rng

from .sel_lexicase_matrix import (
    case_subset,
    fitness_case_matrix,
    require_population,
    validate_case_matrix,
)

__all__: list[str] = ["sel_team"]

_SOLVE_ATOL = 1e-12


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


def _solve_columns(matrix: numpy.ndarray, subset: list[int]) -> numpy.ndarray:
    columns = list(dict.fromkeys(subset))
    if not columns:
        return numpy.zeros((matrix.shape[0], 0), dtype=bool)
    return numpy.isclose(matrix[:, columns], 0.0, atol=_SOLVE_ATOL)


def _next_member(taken: numpy.ndarray, solve: numpy.ndarray, uncovered: numpy.ndarray) -> int:
    if uncovered.size == 0:
        marginal = numpy.zeros(len(taken), dtype=numpy.intp)
    else:
        marginal = numpy.count_nonzero(solve & uncovered, axis=1)
    scores = numpy.where(taken, -1, marginal)
    ties = numpy.flatnonzero(scores == scores.max())
    return int(rng.choice(ties.tolist()))


def sel_team(
    individuals: list[Individual],
    sel_count: int,
    *,
    cases: Sequence[int] | None = None,
    matrix: numpy.ndarray | None = None,
    trust_matrix: bool = False,
) -> list[Individual]:
    """Select a team by greedy maximum coverage of solved fitness cases.

    A case is solved when its value is within ``1e-12`` of zero. Each
    added member is an unused pool individual that covers the most
    still-uncovered cases. Ties are broken at random. Member
    ``fitness`` is not rewritten; score the team on the caller.

    Args:
        individuals: Individuals to select from.
        sel_count: Team size. Values above the pool length return the
            whole pool. ``sel_count <= 0`` returns an empty list.
        cases: Fitness-case indices to cover. All distinct cases are
            used when omitted. Duplicate indices are covered once.
        matrix: Optional ``(n_individuals, n_cases)`` case matrix.
            When omitted, values are read from ``fitness.values``.
        trust_matrix: When ``True``, ``matrix`` is accepted on shape
            alone. Defaults to ``False``.

    Returns:
        Distinct pool members in greedy-add order. ``sel_count == 1``
        is the individual that solves the most selected cases.

    Raises:
        IndexError: If the population is empty or a case index is
            outside the fitness length.
        ValueError: If ``matrix`` shape or values do not match fitness.
    """
    if sel_count <= 0:
        return []
    require_population(individuals)
    packed = _resolve_matrix(individuals, matrix, trust_matrix=trust_matrix)
    solve = _solve_columns(packed, case_subset(individuals, cases))
    taken = numpy.zeros(len(individuals), dtype=bool)
    uncovered = numpy.ones(solve.shape[1], dtype=bool)
    team: list[Individual] = []
    for _ in range(min(sel_count, len(individuals))):
        idx = _next_member(taken, solve, uncovered)
        team.append(individuals[idx])
        taken[idx] = True
        if uncovered.size:
            uncovered &= ~solve[idx]
    return team

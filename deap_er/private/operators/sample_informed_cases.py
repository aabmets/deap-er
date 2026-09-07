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

from collections.abc import Callable
from math import isclose
from numbers import Integral
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual
from deap_er.private.various.rng import rng

from .sel_lexicase_matrix import validate_case_matrix

__all__: list[str] = ["sample_informed_cases"]

type CaseSolved = Callable[[Individual, int], bool]


def _default_solved(individual: Individual, case: int) -> bool:
    return isclose(individual.fitness.values[case], 0.0, abs_tol=1e-12)


def _solve_matrix(individuals: list[Individual], n_cases: int, solved: CaseSolved) -> numpy.ndarray:
    matrix = numpy.empty((n_cases, len(individuals)), dtype=bool)
    for i, individual in enumerate(individuals):
        values = individual.fitness.values
        if len(values) != n_cases:
            raise ValueError("every individual must have a valid fitness of the same length")
        for case in range(n_cases):
            matrix[case, i] = solved(individual, case)
    return matrix


def _hamming(left: numpy.ndarray, right: numpy.ndarray) -> int:
    return int(numpy.count_nonzero(left != right))


def _farthest_first_cases(solve: numpy.ndarray, case_count: int) -> list[int]:
    remaining = list(range(solve.shape[0]))
    first = rng.choice(remaining)
    chosen = [first]
    remaining.remove(first)
    min_dist = {case: _hamming(solve[case], solve[first]) for case in remaining}
    while len(chosen) < case_count:
        max_dist = max(min_dist.values())
        ties = [case for case in remaining if min_dist[case] == max_dist]
        nxt = rng.choice(ties)
        chosen.append(nxt)
        remaining.remove(nxt)
        del min_dist[nxt]
        added = solve[nxt]
        for case in remaining:
            dist = _hamming(solve[case], added)
            if dist < min_dist[case]:
                min_dist[case] = dist
    return chosen


def _solve_from_matrix(matrix: numpy.ndarray) -> numpy.ndarray:
    return numpy.isclose(matrix.T, 0.0, atol=1e-12)


def sample_informed_cases(
    individuals: list[Individual],
    case_count: int,
    *,
    solved: CaseSolved | None = None,
    matrix: numpy.ndarray | None = None,
) -> list[int]:
    """Build a down-sample that prefers distinct fitness cases.

    Two cases are synonymous when the same individuals solve them.
    Distance is the Hamming distance of those solve vectors. A random
    first case is kept, then farthest-first traversal adds the case
    farthest from the nearest already-chosen case. Ties, including a
    tail of zero distances, are broken at random.

    A case is solved when ``fitness.values[case]`` is within ``1e-12``
    of zero. Maximize-only scores and larger residuals need
    ``solved``.

    Args:
        individuals: Population whose fitness vectors supply solve bits.
            Pass a fully scored parent sample if evaluation is sparse.
        case_count: Number of case indices to return. Values above the
            number of cases are capped. ``case_count <= 0`` returns
            an empty list.
        solved: Predicate ``(individual, case) -> bool``. Optional.
            When not the default zero test, ``matrix`` is ignored.
        matrix: Optional ``(n_individuals, n_cases)`` case matrix.
            Used only with the default ``solved`` predicate.

    Returns:
        Distinct fitness-case indices, in the order they were picked.

    Raises:
        ValueError: If ``individuals`` is empty or ``case_count`` is
            not an integer.
        ValueError: If an individual has a missing or mismatched
            fitness length.
    """
    if isinstance(case_count, bool) or not isinstance(case_count, Integral):
        raise ValueError("case_count must be an int")
    count = int(case_count)
    if not individuals:
        raise ValueError("individuals must be non-empty")
    if count <= 0:
        return []
    n_cases = len(individuals[0].fitness.values)
    if n_cases == 0:
        raise ValueError("every individual must have a valid fitness of the same length")
    size = min(count, n_cases)
    predicate = solved if solved is not None else _default_solved
    if matrix is not None and predicate is _default_solved:
        validate_case_matrix(matrix, individuals)
        solve = _solve_from_matrix(matrix)
    else:
        solve = _solve_matrix(individuals, n_cases, predicate)
    return _farthest_first_cases(solve, size)

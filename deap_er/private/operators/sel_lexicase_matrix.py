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

__all__: list[str] = [
    "SOLVE_ATOL",
    "case_index",
    "case_subset",
    "fitness_case_matrix",
    "require_evaluated",
    "require_population",
    "resolve_case_matrix",
    "resolve_case_weights",
    "solve_mask",
    "validate_case_matrix",
]

SOLVE_ATOL = 1e-12


def require_population(individuals: list[Individual]) -> None:
    """Require a non-empty population, matching legacy selector errors.

    Args:
        individuals: Candidate pool.

    Raises:
        IndexError: If ``individuals`` is empty.
    """
    if not individuals:
        raise IndexError("list index out of range")


def require_evaluated(individuals: list[Individual]) -> None:
    """Require every individual to carry a valid fitness.

    A fitness cannot have zero objectives, so empty ``fitness.values``
    means the individual was never evaluated.

    Args:
        individuals: Candidate pool.

    Raises:
        ValueError: If any fitness is not valid.
    """
    if any(len(individual.fitness.values) == 0 for individual in individuals):
        raise ValueError("every individual must have a valid fitness of the same length")


def case_index(idx: object, n_obj: int) -> int:
    """Validate one fitness-case index.

    Args:
        idx: Candidate case index.
        n_obj: Number of fitness cases.

    Returns:
        The index as ``int``.

    Raises:
        IndexError: If ``idx`` is not a valid case index.
    """
    if isinstance(idx, bool) or not isinstance(idx, Integral):
        raise IndexError(f"case index {idx} is out of range for {n_obj} fitness cases")
    value = int(idx)
    if value < 0 or value >= n_obj:
        raise IndexError(f"case index {value} is out of range for {n_obj} fitness cases")
    return value


def case_subset(
    individuals: list[Individual],
    cases: Sequence[int] | None,
    *,
    n_cases: int | None = None,
) -> list[int]:
    """Resolve the case indices used for one lexicase draw.

    Args:
        individuals: Population supplying fitness length.
        cases: Explicit subset, or ``None`` for every case.
        n_cases: Case count to validate against. Defaults to
            ``len(fitness.values)``.

    Returns:
        Case indices in caller order.

    Raises:
        IndexError: If the population is empty or a case index is invalid.
    """
    require_population(individuals)
    bound = n_cases if n_cases is not None else len(individuals[0].fitness.values)
    if cases is None:
        return list(range(bound))
    return [case_index(idx, bound) for idx in cases]


def resolve_case_weights(
    individuals: list[Individual],
    matrix: numpy.ndarray,
    fit_weights: Sequence[float] | None,
) -> tuple[float, ...]:
    """Resolve per-column lexicase signs for a packed case matrix.

    Args:
        individuals: Population the matrix describes.
        matrix: ``(n_individuals, n_cases)`` case matrix.
        fit_weights: Optional signs with one entry per matrix column.

    Returns:
        Maximize/minimize signs aligned with ``matrix`` columns.

    Raises:
        ValueError: If ``fit_weights`` is missing while the matrix
            width differs from ``fitness.values``, or if its length
            does not match ``matrix.shape[1]``.
    """
    require_population(individuals)
    n_fitness = len(individuals[0].fitness.values)
    n_matrix = int(matrix.shape[1])
    if n_matrix == 0:
        if fit_weights is not None and len(fit_weights) != 0:
            raise ValueError("fit_weights must be empty when matrix has no columns")
        return ()
    if fit_weights is None:
        if n_matrix != n_fitness:
            raise ValueError(
                f"matrix has {n_matrix} columns but fitness has {n_fitness}; "
                "pass fit_weights= with one sign per matrix column"
            )
        resolved = tuple(float(weight) for weight in individuals[0].fitness.weights)
    else:
        resolved = tuple(float(weight) for weight in fit_weights)
    if len(resolved) != n_matrix:
        raise ValueError(f"fit_weights must have length {n_matrix}, got {len(resolved)}")
    return resolved


def fitness_case_matrix(individuals: list[Individual]) -> numpy.ndarray:
    """Pack ``fitness.values`` into a dense ``(n_individuals, n_cases)`` matrix.

    Args:
        individuals: Evaluated population. Zero-case fitness is packed
            as ``(n_individuals, 0)``.

    Returns:
        Case values with one row per individual.

    Raises:
        ValueError: If ``individuals`` is empty or fitness lengths differ.
    """
    if not individuals:
        raise ValueError("individuals must be non-empty")
    n_cases = len(individuals[0].fitness.values)
    matrix = numpy.empty((len(individuals), n_cases), dtype=numpy.float64)
    for row, individual in enumerate(individuals):
        values = individual.fitness.values
        if len(values) != n_cases:
            raise ValueError("every individual must have a valid fitness of the same length")
        if n_cases:
            matrix[row] = values
    return matrix


def validate_case_matrix(
    matrix: numpy.ndarray,
    individuals: list[Individual],
    *,
    trust: bool = False,
) -> None:
    """Check that ``matrix`` matches ``individuals`` fitness.

    Args:
        matrix: Pre-packed case matrix.
        individuals: Population the matrix describes.
        trust: When ``True``, only the row count is checked.

    Raises:
        ValueError: If the shape or values do not match ``fitness.values``.
    """
    if not individuals:
        raise ValueError("individuals must be non-empty")
    if trust:
        if matrix.ndim != 2 or matrix.shape[0] != len(individuals):
            raise ValueError(
                f"matrix must have shape ({len(individuals)}, n_cases), got {matrix.shape}"
            )
        return
    n_cases = len(individuals[0].fitness.values)
    expected_shape = (len(individuals), n_cases)
    if matrix.shape != expected_shape:
        raise ValueError(f"matrix must have shape {expected_shape}, got {matrix.shape}")
    for row, individual in enumerate(individuals):
        values = individual.fitness.values
        if len(values) != n_cases:
            raise ValueError("every individual must have a valid fitness of the same length")
        if n_cases and not numpy.array_equal(matrix[row], values):
            raise ValueError("matrix does not match fitness.values")


def resolve_case_matrix(
    individuals: list[Individual],
    matrix: numpy.ndarray | None,
    *,
    trust_matrix: bool,
) -> numpy.ndarray:
    """Return a validated caller matrix, or pack one from fitness.

    Args:
        individuals: Population the matrix describes.
        matrix: Optional pre-packed ``(n_individuals, n_cases)`` matrix.
        trust_matrix: When ``True``, ``matrix`` is accepted on shape alone.

    Returns:
        The case matrix to filter on.

    Raises:
        ValueError: If ``matrix`` shape or values do not match fitness.
    """
    if matrix is None:
        return fitness_case_matrix(individuals)
    validate_case_matrix(matrix, individuals, trust=trust_matrix)
    return matrix


def solve_mask(matrix: numpy.ndarray) -> numpy.ndarray:
    """Mark case values within ``SOLVE_ATOL`` of zero as solved.

    Args:
        matrix: Case values of any shape.

    Returns:
        Boolean array of the same shape.
    """
    return numpy.isclose(matrix, 0.0, rtol=0.0, atol=SOLVE_ATOL)

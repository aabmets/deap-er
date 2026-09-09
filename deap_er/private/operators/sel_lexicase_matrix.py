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

from .epsilon_lexicase_slack import (
    LexicaseMode,
    apply_epsilon_filter,
    apply_strict_filter,
    epsilon_mode_uses_pool_elite,
    slack_for_case,
)

__all__: list[str] = [
    "LexicaseMode",
    "case_index",
    "case_subset",
    "fitness_case_matrix",
    "lexicase_select_vectorized",
    "require_population",
    "validate_case_matrix",
]


def require_population(individuals: list[Individual]) -> None:
    """Require a non-empty population, matching legacy selector errors.

    Args:
        individuals: Candidate pool.

    Raises:
        IndexError: If ``individuals`` is empty.
    """
    if not individuals:
        raise IndexError("list index out of range")


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


def case_subset(individuals: list[Individual], cases: Sequence[int] | None) -> list[int]:
    """Resolve the case indices used for one lexicase draw.

    Args:
        individuals: Population supplying fitness length.
        cases: Explicit subset, or ``None`` for every case.

    Returns:
        Case indices in caller order.

    Raises:
        IndexError: If the population is empty or a case index is invalid.
    """
    require_population(individuals)
    n_obj = len(individuals[0].fitness.values)
    if cases is None:
        return list(range(n_obj))
    return [case_index(idx, n_obj) for idx in cases]


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


def _choice_from_survivors(
    individuals: list[Individual],
    survivors: numpy.ndarray,
) -> Individual:
    if survivors.size == 0:
        return rng.choice(individuals)
    return rng.choice([individuals[i] for i in survivors])


def lexicase_select_vectorized(
    individuals: list[Individual],
    sel_count: int,
    matrix: numpy.ndarray,
    subset: list[int],
    fit_weights: tuple[float, ...],
    *,
    mode: LexicaseMode = "strict",
    epsilon: float | None = None,
) -> list[Individual]:
    """Select individuals by vectorized lexicase filtering.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.
        matrix: Case matrix with shape ``(len(individuals), n_cases)``.
        subset: Fitness-case indices to filter on.
        fit_weights: Per-case maximize/minimize signs from fitness.
        mode: ``strict``, population MAD (``epsilon_auto`` /
            ``epsilon_static``), semi-dynamic pool elite
            (``epsilon_semi``), dynamic pool MAD and elite
            (``epsilon_dynamic``), or fixed slack (``epsilon_fixed``).
        epsilon: Fixed slack when ``mode`` is ``epsilon_fixed``.

    Returns:
        The selected individuals.
    """
    if sel_count <= 0:
        return []
    pool_elite = epsilon_mode_uses_pool_elite(mode)
    selected: list[Individual] = []
    for _ in range(sel_count):
        order = list(subset)
        rng.shuffle(order)
        active = numpy.ones(len(individuals), dtype=bool)
        for case in order:
            if active.sum() <= 1:
                break
            col = matrix[:, case]
            maximize = fit_weights[case] > 0
            if mode == "strict":
                active = apply_strict_filter(active, col, maximize)
            else:
                slack = slack_for_case(col, active, mode, epsilon)
                active = apply_epsilon_filter(
                    active,
                    col,
                    maximize,
                    slack,
                    pool_elite=pool_elite,
                )
        survivors = numpy.flatnonzero(active)
        selected.append(_choice_from_survivors(individuals, survivors))
    return selected

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
from numbers import Integral
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual
from deap_er.private.various.rng import rng

__all__: list[str] = ["lexicase_select", "sel_lexicase", "sel_epsilon_lexicase"]


def _case_index(idx: object, n_obj: int) -> int:
    if isinstance(idx, bool) or not isinstance(idx, Integral):
        raise IndexError(f"case index {idx} is out of range for {n_obj} fitness cases")
    value = int(idx)
    if value < 0 or value >= n_obj:
        raise IndexError(f"case index {value} is out of range for {n_obj} fitness cases")
    return value


def _case_subset(individuals: list[Individual], cases: Sequence[int] | None) -> list[int]:
    n_obj = len(individuals[0].fitness.values)
    if cases is None:
        return list(range(n_obj))
    return [_case_index(idx, n_obj) for idx in cases]


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
        IndexError: If a case index is outside the fitness length.
    """
    if sel_count <= 0:
        return []
    subset = _case_subset(individuals, cases)
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
        choice = rng.choice(candidates)
        selected.append(choice)
    return selected


def sel_lexicase(
    individuals: list[Individual], sel_count: int, *, cases: Sequence[int] | None = None
) -> list[Individual]:
    """Select individuals by lexicase filtering of fitness cases.

    Each selected individual is the last remaining candidate after
    fitness cases are considered one at a time in random order.
    Pass ``cases`` to restrict the filter to a per-generation subset.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.
        cases: Fitness-case indices to filter on. All cases are used
            when omitted. Rebuild the subset each generation; do not
            freeze it on the toolbox.

    Returns:
        The selected individuals.

    Raises:
        IndexError: If a case index is outside the fitness length.
    """

    def keep(candidates: list[Individual], case: int, maximize: bool) -> list[Individual]:
        fn = max if maximize else min
        f_vals = [x.fitness.values[case] for x in candidates]
        best_val = fn(f_vals)
        return [x for x in candidates if x.fitness.values[case] == best_val]

    return lexicase_select(individuals, sel_count, keep, cases=cases)


def sel_epsilon_lexicase(
    individuals: list[Individual],
    sel_count: int,
    epsilon: float | None = None,
    *,
    cases: Sequence[int] | None = None,
) -> list[Individual]:
    """Select individuals by epsilon-lexicase filtering of fitness cases.

    Each selected individual is the last remaining candidate after
    fitness cases are considered one at a time in random order.
    Candidates within ``epsilon`` of the best case value are kept.
    Pass ``cases`` to restrict the filter to a per-generation subset.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.
        epsilon: Slack around the best case value. If omitted, it is
            computed from the median absolute deviation of the case
            values, separately for every case.
        cases: Fitness-case indices to filter on. All cases are used
            when omitted. Rebuild the subset each generation; do not
            freeze it on the toolbox.

    Returns:
        The selected individuals.

    Raises:
        IndexError: If a case index is outside the fitness length.
    """

    def keep(candidates: list[Individual], case: int, maximize: bool) -> list[Individual]:
        errors = [x.fitness.values[case] for x in candidates]
        if epsilon is None:
            median = float(numpy.median(errors))
            slack = float(numpy.median([abs(x - median) for x in errors]))
        else:
            slack = epsilon
        if maximize:
            min_val = max(errors) - slack
            return [x for x in candidates if x.fitness.values[case] >= min_val]
        max_val = min(errors) + slack
        return [x for x in candidates if x.fitness.values[case] <= max_val]

    return lexicase_select(individuals, sel_count, keep, cases=cases)

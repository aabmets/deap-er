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
from collections.abc import Callable

import numpy as np

from deap_er.base.dtypes import Individual
from deap_er.rng import rng

__all__ = ["sel_lexicase", "sel_epsilon_lexicase"]


def _lexicase_select(
    individuals: list[Individual],
    sel_count: int,
    keep: Callable[[list[Individual], int, bool], list[Individual]],
) -> list[Individual]:
    """Select individuals by filtering fitness cases one at a time.

    Cases are considered in a fresh random order for each selection.
    The last remaining candidate wins; ties are broken at random.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.
        keep: Callable that receives the current candidates, the index
            of the active fitness case, and whether that case is
            maximized, and returns the surviving candidates.

    Returns:
        The selected individuals.
    """
    selected = []
    for _i in range(sel_count):
        fit_weights = individuals[0].fitness.weights
        cases = list(range(len(individuals[0].fitness.values)))
        rng.shuffle(cases)
        candidates = individuals
        while len(cases) > 0 and len(candidates) > 1:
            candidates = keep(candidates, cases[0], fit_weights[cases[0]] > 0)
            cases.pop(0)
        choice = rng.choice(candidates)
        selected.append(choice)
    return selected


def sel_lexicase(individuals: list[Individual], sel_count: int) -> list[Individual]:
    """Select individuals by lexicase filtering of fitness cases.

    Each selected individual is the last remaining candidate after
    fitness cases are considered one at a time in random order.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.

    Returns:
        The selected individuals.
    """

    def keep(candidates: list[Individual], case: int, maximize: bool) -> list[Individual]:
        fn = max if maximize else min
        f_vals = [x.fitness.values[case] for x in candidates]
        best_val = fn(f_vals)
        return [x for x in candidates if x.fitness.values[case] == best_val]

    return _lexicase_select(individuals, sel_count, keep)


def sel_epsilon_lexicase(
    individuals: list[Individual], sel_count: int, epsilon: float | None = None
) -> list[Individual]:
    """Select individuals by epsilon-lexicase filtering of fitness cases.

    Each selected individual is the last remaining candidate after
    fitness cases are considered one at a time in random order.
    Candidates within ``epsilon`` of the best case value are kept.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.
        epsilon: Slack around the best case value. If omitted, it is
            computed from the median absolute deviation of the case
            values, separately for every case.

    Returns:
        The selected individuals.
    """

    def keep(candidates: list[Individual], case: int, maximize: bool) -> list[Individual]:
        errors = [x.fitness.values[case] for x in candidates]
        if epsilon is None:
            median = float(np.median(errors))
            slack = float(np.median([abs(x - median) for x in errors]))
        else:
            slack = epsilon
        if maximize:
            min_val = max(errors) - slack
            return [x for x in candidates if x.fitness.values[case] >= min_val]
        max_val = min(errors) + slack
        return [x for x in candidates if x.fitness.values[case] <= max_val]

    return _lexicase_select(individuals, sel_count, keep)

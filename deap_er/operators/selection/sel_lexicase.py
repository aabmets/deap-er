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
import random

import numpy as np

from deap_er.base.dtypes import Individual

__all__ = ["sel_lexicase", "sel_epsilon_lexicase"]


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
    selected = []
    for _i in range(sel_count):
        fit_weights = individuals[0].fitness.weights
        candidates = individuals
        cases = list(range(len(individuals[0].fitness.values)))
        random.shuffle(cases)
        while len(cases) > 0 and len(candidates) > 1:
            fn = min
            if fit_weights[cases[0]] > 0:
                fn = max
            f_vals = [x.fitness.values[cases[0]] for x in candidates]
            best_val = fn(f_vals)
            candidates = [x for x in candidates if x.fitness.values[cases[0]] == best_val]
            cases.pop(0)
        choice = random.choice(candidates)
        selected.append(choice)
    return selected


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
            values.

    Returns:
        The selected individuals.
    """
    selected = []
    for _i in range(sel_count):
        fit_weights = individuals[0].fitness.weights
        cases = list(range(len(individuals[0].fitness.values)))
        random.shuffle(cases)
        candidates = individuals
        while len(cases) > 0 and len(candidates) > 1:
            errors = [x.fitness.values[cases[0]] for x in candidates]
            if not epsilon:
                median = float(np.median(errors))
                epsilon = float(np.median([abs(x - median) for x in errors]))
            if fit_weights[cases[0]] > 0:
                best_val = max(errors)
                min_val = best_val - epsilon
                candidates = [x for x in candidates if x.fitness.values[cases[0]] >= min_val]
            else:
                best_val = min(errors)
                max_val = best_val + epsilon
                candidates = [x for x in candidates if x.fitness.values[cases[0]] <= max_val]
            cases.pop(0)
        choice = random.choice(candidates)
        selected.append(choice)
    return selected

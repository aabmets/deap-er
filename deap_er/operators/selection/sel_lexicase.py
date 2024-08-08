#
#   MIT License
#   
#   Copyright (c) 2022, Mattias Aabmets
#   
#   The contents of this file are subject to the terms and conditions defined in the License.
#   You may not use, modify, or distribute this file except in compliance with the License.
#   
#   SPDX-License-Identifier: MIT
#
import numpy as np
import random


__all__ = ['sel_lexicase', 'sel_epsilon_lexicase']


# ====================================================================================== #
def sel_lexicase(individuals: list, sel_count: int) -> list:
    """
    Returns an individual that does the best on the fitness
    cases when considered one at a time in random order.

    :param individuals: A list of individuals to select from.
    :param sel_count: The number of individuals to select.
    :return: A list of selected individuals.
    """
    selected = []
    for i in range(sel_count):
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


# -------------------------------------------------------------------------------------- #
def sel_epsilon_lexicase(individuals: list, sel_count: int,
                         epsilon: float = None) -> list:
    """
    Returns an individual that does the best on the fitness
    cases when considered one at a time in random order.

    :param individuals: A list of individuals to select from.
    :param sel_count: The number of individuals to select.
    :param epsilon: The epsilon parameter, optional. If not
        provided, the epsilon parameter is automatically
        calculated from the median of fitness values.
    :return: A list of selected individuals.
    """
    selected = []
    for i in range(sel_count):
        fit_weights = individuals[0].fitness.weights
        cases = list(range(len(individuals[0].fitness.values)))
        random.shuffle(cases)
        candidates = individuals
        while len(cases) > 0 and len(candidates) > 1:
            errors = [x.fitness.values[cases[0]] for x in candidates]
            if not epsilon:
                median = np.median(errors)
                epsilon = np.median([abs(x - median) for x in errors])
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

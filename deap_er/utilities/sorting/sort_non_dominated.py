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
from collections import defaultdict

from deap_er.base.dtypes import Individual

__all__ = ["sort_non_dominated"]


def sort_non_dominated(
    individuals: list[Individual], sel_count: int, ffo: bool = False
) -> list[list[Individual]]:
    """Sort individuals into non-dominated Pareto fronts.

    Uses the Fast Non-dominated Sorting Approach. Only the first
    ``sel_count`` individuals are placed into fronts.

    Args:
        individuals: Individuals to sort.
        sel_count: Number of individuals to select.
        ffo: If True, return only the first front. Optional.

    Returns:
        A list of Pareto fronts. The first element is the true
        Pareto front. An empty list if ``sel_count`` is 0.
        When ``ffo`` is True, the list contains only the first front.
    """
    if sel_count == 0:
        return []

    map_fit_ind = defaultdict(list)
    for ind in individuals:
        map_fit_ind[ind.fitness].append(ind)
    fits = list(map_fit_ind.keys())

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)

    for i, fit_i in enumerate(fits):
        for fit_j in fits[i + 1 :]:
            if fit_i.dominates(fit_j):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif fit_j.dominates(fit_i):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    fronts = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])

    if not ffo:
        big_n = min(len(individuals), sel_count)
        while pareto_sorted < big_n:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

    return fronts

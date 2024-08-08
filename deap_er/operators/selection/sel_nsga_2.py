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
from deap_er.utilities.sorting import *
from .sel_helpers import assign_crowding_dist
from operator import attrgetter
from itertools import chain


__all__ = ['sel_nsga_2']


# ====================================================================================== #
def sel_nsga_2(individuals: list, sel_count: int,
               sorting: str = 'standard') -> list:
    """
    Selects the next generation of individuals using the NSGA-II algorithm.
    Usually, the size of **individuals** should be larger than the **sel_count**
    parameter. If the size of **individuals** is equal to **sel_count**, the
    population will be sorted according to their pareto fronts.

    :param individuals: A list of individuals to select from.
    :param sel_count: The number of individuals to select.
    :param sorting: The algorithm to use for non-dominated
        sorting. Can be either 'log' or 'standard' string literal.
    :return: A list of selected individuals.
    """
    if sorting == 'standard':
        pareto_fronts = sort_non_dominated(individuals, sel_count)
    elif sorting == 'log':
        pareto_fronts = sort_log_non_dominated(individuals, sel_count)
    else:
        raise RuntimeError(
            f'selNSGA2: The choice of non-dominated '
            f'sorting method \'{sorting}\' is invalid.'
        )

    for front in pareto_fronts:
        assign_crowding_dist(front)

    chosen = list(chain(*pareto_fronts[:-1]))
    sel_count = sel_count - len(chosen)
    if sel_count > 0:
        attr = attrgetter("fitness.crowding_dist")
        sorted_front = sorted(pareto_fronts[-1], key=attr, reverse=True)
        chosen.extend(sorted_front[:sel_count])

    return chosen

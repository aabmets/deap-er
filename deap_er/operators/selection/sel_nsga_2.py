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
from deap_er.utilities.sorting import *
from deap_er.base.dtypes import Individual
from .sel_helpers import assign_crowding_dist
from operator import attrgetter
from itertools import chain


__all__ = ["sel_nsga_2"]


def sel_nsga_2(
    individuals: list[Individual], sel_count: int, sorting: str = "standard"
) -> list[Individual]:
    """Select the next generation with NSGA-II.

    The pool is usually larger than ``sel_count``. If the two sizes
    are equal, the population is sorted by Pareto front.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.
        sorting: Non-dominated sorting algorithm. Either ``'log'``
            or ``'standard'``.

    Returns:
        The selected individuals.

    Raises:
        RuntimeError: If ``sorting`` is not ``'log'`` or ``'standard'``.
    """
    if sorting == "standard":
        pareto_fronts = sort_non_dominated(individuals, sel_count)
    elif sorting == "log":
        pareto_fronts = sort_log_non_dominated(individuals, sel_count)
    else:
        raise RuntimeError(
            f"selNSGA2: The choice of non-dominated sorting method '{sorting}' is invalid."
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

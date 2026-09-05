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

from itertools import chain
from operator import attrgetter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual
from deap_er.private.various.sort_non_dominated import sort_non_dominated

from .sel_helpers import assign_crowding_dist

__all__: list[str] = ["sel_nsga_2"]


def sel_nsga_2(individuals: list[Individual], sel_count: int) -> list[Individual]:
    """Select the next generation with NSGA-II.

    The pool is usually larger than ``sel_count``. If the two sizes
    are equal, the population is sorted by Pareto front.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.

    Returns:
        The selected individuals.
    """
    if not individuals or sel_count <= 0:
        return []
    pareto_fronts = sort_non_dominated(individuals, sel_count)

    for front in pareto_fronts:
        assign_crowding_dist(front)

    chosen = list(chain(*pareto_fronts[:-1]))
    sel_count = sel_count - len(chosen)
    if sel_count > 0:
        attr = attrgetter("fitness.crowding_dist")
        sorted_front = sorted(pareto_fronts[-1], key=attr, reverse=True)
        chosen.extend(sorted_front[:sel_count])

    return chosen

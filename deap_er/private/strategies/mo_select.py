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

from typing import TYPE_CHECKING, Any

import numpy

from deap_er.private.various.least_contrib import least_contrib
from deap_er.private.various.sort_non_dominated import sort_non_dominated

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = ["select"]


def select(
    strategy: Any, candidates: list[Individual]
) -> tuple[list[Individual], list[Individual]]:
    """Split candidates into ``survivors`` chosen and the remainder.

    Uses non-dominated sorting. When a front would overflow
    ``survivors``, extra members are dropped by least hypervolume
    contribution.

    Args:
        strategy: Multi-objective CMA strategy.
        candidates: Individuals to rank.

    Returns:
        Chosen individuals and those not selected.
    """
    if len(candidates) <= strategy.mu:
        return candidates, []

    pareto_fronts = sort_non_dominated(candidates, len(candidates))

    chosen: list[Individual] = []
    mid_front: list[Individual] = []
    not_chosen: list[Individual] = []

    full = False
    for front in pareto_fronts:
        if len(chosen) + len(front) <= strategy.mu and not full:
            chosen += front
        elif not mid_front and len(chosen) < strategy.mu:
            mid_front = front
            full = True
        else:
            not_chosen += front

    k = strategy.mu - len(chosen)
    if k > 0 and mid_front:
        ref = numpy.max(numpy.array([ind.fitness.wvalues for ind in candidates]) * -1, axis=0) + 1

        for _ in range(len(mid_front) - k):
            idx = least_contrib(mid_front, ref)
            not_chosen.append(mid_front.pop(idx))

        chosen += mid_front

    return chosen, not_chosen

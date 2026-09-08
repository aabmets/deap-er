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

from collections import defaultdict
from typing import TYPE_CHECKING

import moocore
import numpy

from deap_er.private.fitness import has_comparable_fitness

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = ["sort_non_dominated"]


def sort_non_dominated(individuals: list[Individual], sel_count: int) -> list[list[Individual]]:
    """Sort individuals into non-dominated Pareto fronts.

    Uses ``moocore.pareto_rank`` on ``fitness.wvalues`` (higher is
    better). Fronts are truncated once they hold at least
    ``sel_count`` individuals. Individuals without a comparable
    fitness (missing, invalid, or non-finite) are ignored.

    Args:
        individuals: Individuals to sort.
        sel_count: Number of individuals to place into fronts.

    Returns:
        A list of Pareto fronts. The first element is the true
        Pareto front. An empty list if ``sel_count`` is not
        positive. A single empty front if no rankable individual
        remains and ``sel_count`` is positive.
    """
    if sel_count <= 0:
        return []
    ranked = [ind for ind in individuals if has_comparable_fitness(ind)]
    if not ranked:
        return [[]]

    points = numpy.array([ind.fitness.wvalues for ind in ranked], dtype=float)
    ranks = moocore.pareto_rank(points, maximise=True)

    by_rank: defaultdict[int, list[Individual]] = defaultdict(list)
    for ind, rank in zip(ranked, ranks, strict=True):
        by_rank[int(rank)].append(ind)

    fronts: list[list[Individual]] = []
    placed = 0
    for rank in range(int(max(ranks)) + 1):
        front = by_rank[rank]
        fronts.append(front)
        placed += len(front)
        if placed >= sel_count:
            break
    return fronts

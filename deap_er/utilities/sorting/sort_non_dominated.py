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

import moocore
import numpy

from deap_er.base.typedefs import Individual

__all__ = ["sort_non_dominated"]


def sort_non_dominated(individuals: list[Individual], sel_count: int) -> list[list[Individual]]:
    """Sort individuals into non-dominated Pareto fronts.

    Uses ``moocore.pareto_rank`` on ``fitness.wvalues`` (higher is
    better). Fronts are truncated once they hold at least
    ``sel_count`` individuals.

    Args:
        individuals: Individuals to sort.
        sel_count: Number of individuals to place into fronts.

    Returns:
        A list of Pareto fronts. The first element is the true
        Pareto front. An empty list if ``sel_count`` is 0. A
        single empty front if ``individuals`` is empty and
        ``sel_count`` is positive.
    """
    if sel_count == 0:
        return []
    if not individuals:
        return [[]]

    points = numpy.array([ind.fitness.wvalues for ind in individuals], dtype=float)
    ranks = moocore.pareto_rank(points, maximise=True)

    by_rank: defaultdict[int, list[Individual]] = defaultdict(list)
    for ind, rank in zip(individuals, ranks, strict=True):
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

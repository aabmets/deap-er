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

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import moocore
import numpy

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = ["sort_non_dominated"]


def _rankable_fitness(individual: Any) -> bool:
    """Return whether ``individual`` has a finite, valid fitness.

    Args:
        individual: Candidate that may lack a fitness attribute.

    Returns:
        True when fitness exists, is valid, and every weighted
        objective is finite.
    """
    if not hasattr(individual, "fitness"):
        return False
    fitness = individual.fitness
    if not fitness.is_valid():
        return False
    return all(math.isfinite(float(value)) for value in fitness.wvalues)


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
    ranked = [ind for ind in individuals if _rankable_fitness(ind)]
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

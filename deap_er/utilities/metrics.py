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
from math import hypot, sqrt
from typing import Any

import numpy

from deap_er.base.dtypes import *


def nsga_diversity(population: list[Individual], first: Individual, last: Individual) -> float:
    """Return the NSGA-II diversity metric of a Pareto front.

    ``population`` is the front to score. ``first`` and ``last`` are
    the extreme points of the optimal Pareto front, as in Deb's
    original NSGA-II article. Smaller values indicate better spread.

    Args:
        population: Pareto front to evaluate.
        first: First extreme point of the optimal Pareto front.
        last: Last extreme point of the optimal Pareto front.

    Returns:
        The diversity metric of the front.
    """
    df = hypot(
        population[0].fitness.values[0] - first[0], population[0].fitness.values[1] - first[1]
    )
    dl = hypot(
        population[-1].fitness.values[0] - last[0], population[-1].fitness.values[1] - last[1]
    )

    def fn(f_: Individual, s_: Individual) -> float:
        return hypot(
            f_.fitness.values[0] - s_.fitness.values[0], f_.fitness.values[1] - s_.fitness.values[1]
        )

    zipper = zip(population[:-1], population[1:], strict=False)
    dt = [fn(first, second) for first, second in zipper]

    if len(population) == 1:
        return df + dl

    dm = sum(dt) / len(dt)
    di = sum(abs(d_i - dm) for d_i in dt)
    delta = (df + dl + di) / (df + dl + len(dt) * dm)
    return delta


def nsga_convergence(population: list[Individual], optimal: list[Individual]) -> float:
    """Return the NSGA-II convergence metric of a Pareto front.

    ``population`` is the front to score and ``optimal`` is the true
    Pareto front, as in Deb's original NSGA-II article. Smaller values
    indicate closer solutions.

    Args:
        population: Pareto front to evaluate.
        optimal: Optimal Pareto front.

    Returns:
        The convergence metric of the front.
    """
    distances = []
    for ind in population:
        distances.append(float("inf"))
        for opt_ind in optimal:
            dist = 0.0
            for i in range(len(opt_ind)):
                dist += (ind.fitness.values[i] - opt_ind[i]) ** 2
            if dist < distances[-1]:
                distances[-1] = dist
        distances[-1] = sqrt(distances[-1])
    return float(sum(distances) / len(distances))


def inv_gen_dist(ind1: Individual, ind2: Individual) -> Any:
    """Compute the inverted generational distance between two point sets.

    IGD measures how well one approximation covers another in
    multi-objective optimization.

    Args:
        ind1: First point set.
        ind2: Second point set.

    Returns:
        The average distance from each point in ``ind2`` to the
        nearest point in ``ind1``.
    """
    from scipy import spatial

    distances = spatial.distance.cdist(list(ind1), list(ind2))
    minima = numpy.min(distances, axis=0)
    return numpy.average(minima)

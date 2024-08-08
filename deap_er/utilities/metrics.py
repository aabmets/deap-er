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
from deap_er.base.dtypes import *
from typing import Optional, Any
from math import hypot, sqrt
import numpy


# ====================================================================================== #
def nsga_diversity(population: list, first: Individual, last: Individual) -> float:
    """
    Given a Pareto front **population** and the two extreme points **first** and
    **last** of the optimal Pareto front, this function returns the diversity metric
    of the **population** as explained in the original NSGA-II article by K. Deb.
    Smaller values indicate better solutions.

    :param population: The Pareto front to be evaluated.
    :param first: The first extreme point of the optimal Pareto front.
    :param last: The last extreme point of the optimal Pareto front.
    :return: The diversity metric of the front.

    :type first: :ref:`Individual <datatypes>`
    :type last: :ref:`Individual <datatypes>`
    """
    df = hypot(
        population[0].fitness.values[0] - first[0],
        population[0].fitness.values[1] - first[1]
    )
    dl = hypot(
        population[-1].fitness.values[0] - last[0],
        population[-1].fitness.values[1] - last[1]
    )

    def fn(f_, s_):
        return hypot(
            f_.fitness.values[0] - s_.fitness.values[0],
            f_.fitness.values[1] - s_.fitness.values[1]
        )
    zipper = zip(population[:-1], population[1:])
    dt = [fn(first, second) for first, second in zipper]

    if len(population) == 1:
        return df + dl

    dm = sum(dt)/len(dt)
    di = sum(abs(d_i - dm) for d_i in dt)
    delta = (df + dl + di) / (df + dl + len(dt) * dm)
    return delta


# -------------------------------------------------------------------------------------- #
def nsga_convergence(population: list, optimal: list) -> float:
    """
    Given a Pareto **front** and the **optimal** Pareto front, this function
    returns the convergence metric of the front as explained in the original
    NSGA-II article by K. Deb. Smaller values indicate more optimal solutions.

    :param population: The Pareto front to be evaluated.
    :param optimal: The optimal Pareto front.
    :return: The convergence metric of the front.
    """
    distances = []
    for ind in population:
        distances.append(float("inf"))
        for opt_ind in optimal:
            dist = 0.
            for i in range(len(opt_ind)):
                dist += (ind.fitness.values[i] - opt_ind[i])**2
            if dist < distances[-1]:
                distances[-1] = dist
        distances[-1] = sqrt(distances[-1])
    return sum(distances) / len(distances)


# -------------------------------------------------------------------------------------- #
def inv_gen_dist(ind1: Individual, ind2: Individual) -> tuple[Any, Optional[Any]]:
    """
    Computes the Inverted Generational Distance (IGD) between the two individuals.
    The IGD is a metric for assessing the quality of approximations to the
    Pareto front obtained by multi-objective optimization algorithms.

    :param ind1: The first individual.
    :param ind2: The second individual.
    :return: The IGD between the two individuals.

    :type ind1: :ref:`Individual <datatypes>`
    :type ind2: :ref:`Individual <datatypes>`
    """
    from scipy import spatial
    distances = spatial.distance.cdist(ind1, ind2)
    minima = numpy.min(distances, axis=0)
    return numpy.average(minima)

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
import math

import numpy

from deap_er.base.typedefs import Individual
from deap_er.rng import rng

__all__: list[str] = []


def _sq_distance(ind_i: Individual, ind_j: Individual, big_l: int) -> float:
    """Return the squared objective-space distance between two individuals.

    Args:
        ind_i: First individual.
        ind_j: Second individual.
        big_l: Number of objectives to compare.

    Returns:
        The squared distance between the two fitness vectors.
    """
    dist = 0.0
    for small_l in range(big_l):
        val = ind_i.fitness.values[small_l] - ind_j.fitness.values[small_l]
        dist += val * val
    return float(dist)


def _raw_fitness(individuals: list[Individual]) -> list[float]:
    """Return the SPEA-II raw fitness of every individual.

    An individual's raw fitness is the sum of the strengths of the
    individuals that dominate it, so non-dominated members score zero.

    Args:
        individuals: Individuals to rank.

    Returns:
        One raw fitness value per individual, in input order.
    """
    if not individuals:
        return []
    wvals = numpy.array([ind.fitness.wvalues for ind in individuals], dtype=float)
    ge = wvals[:, numpy.newaxis, :] >= wvals[numpy.newaxis, :, :]
    gt = wvals[:, numpy.newaxis, :] > wvals[numpy.newaxis, :, :]
    dominates = ge.all(axis=2) & gt.any(axis=2)
    strength = dominates.sum(axis=1, dtype=float)
    fits = (strength @ dominates).tolist()
    return [float(value) for value in fits]


def _fill_from_density(
    individuals: list[Individual], chosen: list[int], fits: list[float], sel_count: int
) -> list[int]:
    """Top up an undersized archive with the least crowded individuals.

    Adds a density term to every raw fitness, then takes the best of
    the individuals that are not already chosen.

    Args:
        individuals: Individuals to select from.
        chosen: Indices of the non-dominated individuals.
        fits: Raw fitness values, modified in place with the density.
        sel_count: Number of individuals to select in total.

    Returns:
        The chosen indices, extended to ``sel_count`` entries.
    """
    big_n = len(individuals)
    big_k = math.sqrt(big_n)
    vals = numpy.array([ind.fitness.values for ind in individuals], dtype=float)
    delta = vals[:, numpy.newaxis, :] - vals[numpy.newaxis, :, :]
    sq_dist = numpy.einsum("ijk,ijk->ij", delta, delta)

    for i in range(big_n):
        distances = [0.0] * big_n
        if i + 1 < big_n:
            distances[i + 1 :] = sq_dist[i, i + 1 :].tolist()
        kth_dist = _randomized_select(distances, 0, big_n - 1, big_k)
        fits[i] += 1.0 / (kth_dist + 2.0)

    chosen_set = set(chosen)
    next_indices = [(fits[i], i) for i in range(big_n) if i not in chosen_set]
    next_indices.sort()
    return chosen + [i for _, i in next_indices[: sel_count - len(chosen)]]


def _partition(array: list[float], begin: int, end: int) -> int:
    """Partition a slice of ``array`` around the value at ``begin``.

    The slice ``array[begin:end + 1]`` is modified in place.

    Args:
        array: Sequence to partition.
        begin: Inclusive start of the slice and initial pivot value.
        end: Inclusive end of the slice.

    Returns:
        Split index of the partitioned slice.
    """
    x = array[begin]
    i = begin - 1
    j = end + 1
    while True:
        j -= 1
        while array[j] > x:
            j -= 1
        i += 1
        while array[i] < x:
            i += 1
        if i < j:
            array[i], array[j] = array[j], array[i]
        else:
            return j


def _randomized_partition(array: list[float], begin: int, end: int) -> int:
    """Partition a slice of ``array`` around a randomly chosen pivot.

    The slice ``array[begin:end + 1]`` is modified in place.

    Args:
        array: Sequence to partition.
        begin: Inclusive start of the slice.
        end: Inclusive end of the slice.

    Returns:
        Split index of the partitioned slice.
    """
    i = rng.randint(begin, end)
    array[begin], array[i] = array[i], array[begin]
    return _partition(array, begin, end)


def _randomized_select(array: list[float], begin: int, end: int, i: float) -> float:
    """Return the element of rank ``i`` in a slice of ``array``.

    The slice ``array[begin:end + 1]`` is modified in place. ``i``
    may be non-integral; the comparison uses it as an order-statistic
    threshold.

    Args:
        array: Sequence to search.
        begin: Inclusive start of the slice.
        end: Inclusive end of the slice.
        i: Desired rank within the slice.

    Returns:
        The selected order statistic.
    """
    if begin == end:
        return array[begin]
    q = _randomized_partition(array, begin, end)
    k = q - begin + 1
    if i < k:
        return _randomized_select(array, begin, q, i)
    else:
        return _randomized_select(array, q + 1, end, i - k)

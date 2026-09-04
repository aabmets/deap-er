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
import random

from deap_er.base.dtypes import Individual

__all__ = ["sel_spea_2"]


def sel_spea_2(individuals: list[Individual], sel_count: int) -> list[Individual]:
    """Select the next generation with SPEA-II.

    The pool is usually larger than ``sel_count``. If the two sizes
    are equal, the population is sorted by Pareto front.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.

    Returns:
        The selected individuals.
    """
    fits = _raw_fitness(individuals)

    chosen = [i for i in range(len(individuals)) if fits[i] < 1]
    if len(chosen) < sel_count:
        chosen = _fill_from_density(individuals, chosen, fits, sel_count)
    elif len(chosen) > sel_count:
        chosen = _truncate_archive(individuals, chosen, sel_count)

    return [individuals[i] for i in chosen]


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
    big_n = len(individuals)
    strength_fits = [0.0] * big_n
    fits = [0.0] * big_n
    dominating_individuals = [list() for _ in range(big_n)]

    for i, ind_i in enumerate(individuals):
        for j, ind_j in enumerate(individuals[i + 1 :], i + 1):
            if ind_i.fitness.dominates(ind_j.fitness):
                strength_fits[i] += 1
                dominating_individuals[j].append(i)
            elif ind_j.fitness.dominates(ind_i.fitness):
                strength_fits[j] += 1
                dominating_individuals[i].append(j)

    for i in range(big_n):
        for j in dominating_individuals[i]:
            fits[i] += strength_fits[j]

    return fits


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
    big_l = len(individuals[0].fitness.values)
    big_n = len(individuals)
    big_k = math.sqrt(big_n)

    for i in range(big_n):
        distances = [0.0] * big_n
        for j in range(i + 1, big_n):
            distances[j] = _sq_distance(individuals[i], individuals[j], big_l)
        kth_dist = _randomized_select(distances, 0, big_n - 1, big_k)
        fits[i] += 1.0 / (kth_dist + 2.0)

    next_indices = [(fits[i], i) for i in range(big_n) if i not in chosen]
    next_indices.sort()
    return chosen + [i for _, i in next_indices[: sel_count - len(chosen)]]


def _truncate_archive(
    individuals: list[Individual], chosen: list[int], sel_count: int
) -> list[int]:
    """Shrink an oversized archive to ``sel_count`` entries.

    Repeatedly drops the individual with the closest neighbour, using
    the next-nearest neighbours to break ties.

    Args:
        individuals: Individuals to select from.
        chosen: Indices of the non-dominated individuals.
        sel_count: Number of individuals to keep.

    Returns:
        The chosen indices, reduced to ``sel_count`` entries.
    """
    big_l = len(individuals[0].fitness.values)
    big_n = len(chosen)

    distances = [[0.0] * big_n for _ in range(big_n)]
    sorted_indices = [[0] * big_n for _ in range(big_n)]
    for i in range(big_n):
        for j in range(i + 1, big_n):
            dist = _sq_distance(individuals[chosen[i]], individuals[chosen[j]], big_l)
            distances[i][j] = dist
            distances[j][i] = dist
        distances[i][i] = -1

    for i in range(big_n):
        for j in range(1, big_n):
            small_l = j
            while small_l > 0 and distances[i][j] < distances[i][sorted_indices[i][small_l - 1]]:
                sorted_indices[i][small_l] = sorted_indices[i][small_l - 1]
                small_l -= 1
            sorted_indices[i][small_l] = j

    size = big_n
    to_remove = []
    while size > sel_count:
        min_pos = _most_crowded(distances, sorted_indices, big_n, size)

        for i in range(big_n):
            distances[i][min_pos] = float("inf")
            distances[min_pos][i] = float("inf")

            for j in range(1, size - 1):
                if sorted_indices[i][j] == min_pos:
                    sorted_indices[i][j] = sorted_indices[i][j + 1]
                    sorted_indices[i][j + 1] = min_pos

        to_remove.append(min_pos)
        size -= 1

    chosen = list(chosen)
    for index in reversed(sorted(to_remove)):
        del chosen[index]
    return chosen


def _most_crowded(
    distances: list[list[float]], sorted_indices: list[list[int]], big_n: int, size: int
) -> int:
    """Return the archive position with the closest neighbours.

    Args:
        distances: Pairwise squared distances between archive members.
        sorted_indices: Neighbour indices of each member, nearest first.
        big_n: Number of archive members.
        size: Number of members still active.

    Returns:
        The position of the most crowded member.
    """
    min_pos = 0
    for i in range(1, big_n):
        for j in range(1, size):
            dist_i_sorted_j = distances[i][sorted_indices[i][j]]
            dist_min_sorted_j = distances[min_pos][sorted_indices[min_pos][j]]

            if dist_i_sorted_j < dist_min_sorted_j:
                min_pos = i
                break
            elif dist_i_sorted_j > dist_min_sorted_j:
                break
    return min_pos


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
    i = random.randint(begin, end)
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

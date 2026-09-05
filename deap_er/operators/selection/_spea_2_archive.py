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
from deap_er.base.typedefs import Individual

from ._spea_2 import _sq_distance

__all__: list[str] = []


def _archive_distance_tables(
    individuals: list[Individual], chosen: list[int]
) -> tuple[list[list[float]], list[list[int]], int]:
    """Build pairwise distances and nearest-neighbour orders.

    Args:
        individuals: Individuals to select from.
        chosen: Indices of the non-dominated individuals.

    Returns:
        Pairwise squared distances, neighbour indices nearest first,
        and the archive size.
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
        _sort_neighbours(distances[i], sorted_indices[i], big_n)
    return distances, sorted_indices, big_n


def _sort_neighbours(row: list[float], order: list[int], big_n: int) -> None:
    """Insertion-sort neighbour indices of one archive member.

    Args:
        row: Squared distances from this member to the archive.
        order: Neighbour-index row to fill, nearest first.
        big_n: Archive size.
    """
    for j in range(1, big_n):
        small_l = j
        while small_l > 0 and row[j] < row[order[small_l - 1]]:
            order[small_l] = order[small_l - 1]
            small_l -= 1
        order[small_l] = j


def _drop_crowded(
    distances: list[list[float]], sorted_indices: list[list[int]], big_n: int, sel_count: int
) -> list[int]:
    """Return archive positions to drop until ``sel_count`` remain.

    Args:
        distances: Pairwise squared distances between archive members.
        sorted_indices: Neighbour indices of each member, nearest first.
        big_n: Number of archive members.
        sel_count: Number of individuals to keep.

    Returns:
        Positions to remove, in the order they were dropped.
    """
    size = big_n
    to_remove = []
    while size > sel_count:
        min_pos = _most_crowded(distances, sorted_indices, big_n, size)
        _invalidate_slot(distances, sorted_indices, min_pos, big_n, size)
        to_remove.append(min_pos)
        size -= 1
    return to_remove


def _invalidate_slot(
    distances: list[list[float]],
    sorted_indices: list[list[int]],
    min_pos: int,
    big_n: int,
    size: int,
) -> None:
    """Mark one archive position as removed.

    Args:
        distances: Pairwise squared distances between archive members.
        sorted_indices: Neighbour indices of each member, nearest first.
        min_pos: Position being dropped.
        big_n: Number of archive members.
        size: Number of members still active before this drop.
    """
    for i in range(big_n):
        distances[i][min_pos] = float("inf")
        distances[min_pos][i] = float("inf")
        for j in range(1, size - 1):
            if sorted_indices[i][j] == min_pos:
                sorted_indices[i][j] = sorted_indices[i][j + 1]
                sorted_indices[i][j + 1] = min_pos


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
    distances, sorted_indices, big_n = _archive_distance_tables(individuals, chosen)
    to_remove = _drop_crowded(distances, sorted_indices, big_n, sel_count)
    chosen = list(chosen)
    for index in sorted(to_remove, reverse=True):
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

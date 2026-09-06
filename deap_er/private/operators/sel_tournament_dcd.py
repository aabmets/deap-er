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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual
from deap_er.private.various.rng import rng

__all__: list[str] = ["sel_tournament_dcd"]


def _dcd_tourney(ind1: Individual, ind2: Individual) -> Individual:
    """Return the better of two individuals by dominance, then crowding.

    Args:
        ind1: First contestant.
        ind2: Second contestant.

    Returns:
        The winning individual.
    """
    if ind1.fitness.dominates(ind2.fitness):
        return ind1
    elif ind2.fitness.dominates(ind1.fitness):
        return ind2
    if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
        return ind2
    elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
        return ind1
    if rng.random() <= 0.5:
        return ind1
    return ind2


def sel_tournament_dcd(individuals: list[Individual], sel_count: int) -> list[Individual]:
    """Select by pairwise dominance, breaking ties with crowding distance.

    When ``sel_count`` is a multiple of four the original paired
    shuffle is used. Other counts run pairwise contests until enough
    winners are collected. Each individual must already have a
    ``crowding_dist`` attribute, which ``assign_crowding_dist`` can
    set.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.

    Returns:
        The selected individuals.

    Raises:
        ValueError: If ``sel_count`` is larger than the pool.
    """
    if sel_count <= 0:
        return []
    if sel_count > len(individuals):
        raise ValueError(
            "sel_tournament_dcd: count must be less than or equal to individuals length."
        )

    if sel_count % 4 == 0:
        individuals_1 = rng.sample(individuals, len(individuals))
        individuals_2 = rng.sample(individuals, len(individuals))

        chosen = []
        for i in range(0, sel_count, 4):
            chosen.append(_dcd_tourney(individuals_1[i], individuals_1[i + 1]))
            chosen.append(_dcd_tourney(individuals_1[i + 2], individuals_1[i + 3]))
            chosen.append(_dcd_tourney(individuals_2[i], individuals_2[i + 1]))
            chosen.append(_dcd_tourney(individuals_2[i + 2], individuals_2[i + 3]))
        return chosen

    if sel_count == 1 and len(individuals) == 1:
        return [individuals[0]]

    chosen = []
    pool = list(individuals)
    while len(chosen) < sel_count:
        rng.shuffle(pool)
        for i in range(0, len(pool) - 1, 2):
            chosen.append(_dcd_tourney(pool[i], pool[i + 1]))
            if len(chosen) >= sel_count:
                break
    return chosen

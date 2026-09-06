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

from collections.abc import Callable
from functools import partial
from operator import attrgetter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual
from deap_er.private.various.rng import rng

from .sel_various import sel_random

__all__: list[str] = ["sel_tournament", "sel_double_tournament", "sel_tournament_dcd"]


def sel_tournament(
    individuals: list[Individual], rounds: int, contestants: int, fit_attr: str = "fitness"
) -> list[Individual]:
    """Select the best of ``contestants`` random individuals, ``rounds`` times.

    Args:
        individuals: Individuals to select from.
        rounds: Number of tournament rounds.
        contestants: Number of individuals in each round.
        fit_attr: Attribute used as the selection criterion.

    Returns:
        The selected individuals.
    """
    if rounds <= 0:
        return []
    n = len(individuals)
    if n == 0:
        raise IndexError("Cannot choose from an empty sequence")
    key = attrgetter(fit_attr)
    if contestants < 1:
        max((), key=key)
    idxs = rng.integers(0, n, size=rounds * contestants)
    chosen: list[Individual] = []
    total = rounds * contestants
    if contestants == 1:
        return [individuals[idx] for idx in idxs]
    if contestants == 2:
        for i in range(0, total, 2):
            first = individuals[idxs[i]]
            second = individuals[idxs[i + 1]]
            chosen.append(second if key(second) > key(first) else first)
        return chosen
    if contestants == 3:
        for i in range(0, total, 3):
            winner = individuals[idxs[i]]
            best = key(winner)
            second = individuals[idxs[i + 1]]
            score = key(second)
            if score > best:
                winner, best = second, score
            third = individuals[idxs[i + 2]]
            if key(third) > best:
                winner = third
            chosen.append(winner)
        return chosen
    for i in range(0, total, contestants):
        winner = individuals[idxs[i]]
        best = key(winner)
        for offset in range(1, contestants):
            candidate = individuals[idxs[i + offset]]
            score = key(candidate)
            if score > best:
                winner, best = candidate, score
        chosen.append(winner)
    return chosen


def sel_double_tournament(
    individuals: list[Individual],
    rounds: int,
    fitness_size: int,
    parsimony_size: float,
    fitness_first: bool,
    fit_attr: str = "fitness",
) -> list[Individual]:
    """Select with a fitness tournament and a size tournament.

    The size contest can be used in genetic programming as a bloat
    control technique.

    Args:
        individuals: Individuals to select from.
        rounds: Number of tournament rounds.
        fitness_size: Number of individuals in each fitness tournament.
        parsimony_size: Number of individuals in each size tournament.
            Must be in ``[1, 2]``.
        fitness_first: If True, run the fitness tournament first.
        fit_attr: Attribute used as the fitness selection criterion.

    Returns:
        The selected individuals.

    Raises:
        ValueError: If ``parsimony_size`` is outside ``[1, 2]``.
    """
    if not (1 <= parsimony_size <= 2):
        raise ValueError("Parsimony tournament size has to be in the range of [1, 2].")

    def _size_tourney(
        pool: list[Individual], sel_count: int, select: Callable[..., Any]
    ) -> list[Individual]:
        """Run the parsimony (size) half of the double tournament.

        Args:
            pool: Individuals to select from.
            sel_count: Number of size contests to run.
            select: Selection callable used to pick the two contestants.

        Returns:
            Winners of the size contests.
        """
        chosen = []
        for _i in range(sel_count):
            prob = parsimony_size / 2.0
            ind1, ind2 = select(pool, sel_count=2)
            if len(ind1) > len(ind2):
                ind1, ind2 = ind2, ind1
            elif len(ind1) == len(ind2):
                prob = 0.5
            chosen.append(ind1 if rng.random() < prob else ind2)
        return chosen

    def _fit_tourney(
        pool: list[Individual], sel_count: int, select: Callable[..., Any]
    ) -> list[Individual]:
        """Run the fitness half of the double tournament.

        Args:
            pool: Individuals to select from.
            sel_count: Number of fitness contests to run.
            select: Selection callable used to pick the contestants.

        Returns:
            Winners of the fitness contests.
        """
        chosen = []
        for _i in range(sel_count):
            aspirants = select(pool, sel_count=fitness_size)
            chosen.append(max(aspirants, key=attrgetter(fit_attr)))
        return chosen

    if fitness_first:
        t_fit = partial(_fit_tourney, select=sel_random)
        return _size_tourney(individuals, rounds, t_fit)
    t_size = partial(_size_tourney, select=sel_random)
    return _fit_tourney(individuals, rounds, t_size)


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

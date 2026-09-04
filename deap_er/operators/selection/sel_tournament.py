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
from collections.abc import Callable
from functools import partial
from operator import attrgetter
from typing import Any

from deap_er.base.typedefs import Individual
from deap_er.rng import rng

from .sel_various import sel_random

__all__ = ["sel_tournament", "sel_double_tournament", "sel_tournament_dcd"]


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
    chosen = []
    for _ in range(rounds):
        aspirants = sel_random(individuals, contestants)
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
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


def sel_tournament_dcd(individuals: list[Individual], sel_count: int) -> list[Individual]:
    """Select by pairwise dominance, breaking ties with crowding distance.

    If ``sel_count`` equals the pool size, that size must be a multiple
    of four. Each individual must already have a ``crowding_dist``
    attribute, which ``assign_crowding_dist`` can set.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.

    Returns:
        The selected individuals.

    Raises:
        ValueError: If ``sel_count`` is larger than the pool, or if
            ``sel_count`` equals the pool size and is not divisible
            by four.
    """
    if sel_count > len(individuals):
        raise ValueError(
            "sel_tournament_dcd: count must be less than or equal to individuals length."
        )

    if sel_count == len(individuals) and sel_count % 4 != 0:
        raise ValueError(
            "sel_tournament_dcd: sel_count must be divisible "
            "by four if sel_count == len(individuals)"
        )

    def tourney(ind1: Individual, ind2: Individual) -> Individual:
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

    individuals_1 = rng.sample(individuals, len(individuals))
    individuals_2 = rng.sample(individuals, len(individuals))

    chosen = []
    for i in range(0, sel_count, 4):
        chosen.append(tourney(individuals_1[i], individuals_1[i + 1]))
        chosen.append(tourney(individuals_1[i + 2], individuals_1[i + 3]))
        chosen.append(tourney(individuals_2[i], individuals_2[i + 1]))
        chosen.append(tourney(individuals_2[i + 2], individuals_2[i + 3]))

    return chosen

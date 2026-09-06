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

from operator import attrgetter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deap_er.private.fitness import Fitness
    from deap_er.private.typedefs import Individual
from deap_er.private.various.rng import rng

__all__: list[str] = [
    "sel_random",
    "sel_best",
    "sel_worst",
    "sel_roulette",
    "sel_stochastic_universal_sampling",
]

_WHEEL_EPS = 1e-12


def _wheel_weight(fitness: Fitness, floor: float) -> float:
    """Return a non-negative wheel slice for ``fitness``.

    Args:
        fitness: Fitness whose first weighted objective is the slice.
        floor: Minimum first-objective wvalue in the pool.

    Returns:
        A non-negative slice width.
    """
    weight = float(fitness.wvalues[0])
    if floor < 0.0:
        return weight - floor + _WHEEL_EPS
    return weight


def sel_random(individuals: list[Individual], sel_count: int) -> list[Individual]:
    """Select ``sel_count`` individuals uniformly at random.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.

    Returns:
        The selected individuals.
    """
    return [rng.choice(individuals) for _ in range(sel_count)]


def sel_best(
    individuals: list[Individual], sel_count: int, fit_attr: str = "fitness"
) -> list[Individual]:
    """Select the ``sel_count`` best individuals.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.
        fit_attr: Attribute used as the selection criterion.

    Returns:
        The selected individuals.
    """
    key = attrgetter(fit_attr)
    return sorted(individuals, key=key, reverse=True)[:sel_count]


def sel_worst(
    individuals: list[Individual], sel_count: int, fit_attr: str = "fitness"
) -> list[Individual]:
    """Select the ``sel_count`` worst individuals.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.
        fit_attr: Attribute used as the selection criterion.

    Returns:
        The selected individuals.
    """
    key = attrgetter(fit_attr)
    return sorted(individuals, key=key)[:sel_count]


def sel_roulette(
    individuals: list[Individual], sel_count: int, fit_attr: str = "fitness"
) -> list[Individual]:
    """Select ``sel_count`` individuals by roulette-wheel sampling.

    Each draw uses only the first weighted objective of ``fit_attr``.
    The returned list holds references to the input individuals.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.
        fit_attr: Attribute used as the selection criterion.

    Returns:
        The selected individuals.
    """
    if sel_count <= 0 or not individuals:
        return []
    key = attrgetter(fit_attr)
    sorted_ = sorted(individuals, key=key, reverse=True)
    floor = min(getattr(ind, fit_attr).wvalues[0] for ind in individuals)
    sum_fits = sum(_wheel_weight(getattr(ind, fit_attr), floor) for ind in individuals)
    if sum_fits == 0:
        return [rng.choice(individuals) for _ in range(sel_count)]
    chosen = []
    for _ in range(sel_count):
        u = rng.random() * sum_fits
        sum_ = 0
        for ind in sorted_:
            sum_ += _wheel_weight(getattr(ind, fit_attr), floor)
            if sum_ > u:
                chosen.append(ind)
                break

    return chosen


def sel_stochastic_universal_sampling(
    individuals: list[Individual], sel_count: int, fit_attr: str = "fitness"
) -> list[Individual]:
    """Select ``sel_count`` individuals by stochastic universal sampling.

    A single random offset samples the wheel at evenly spaced
    intervals. Only the first weighted objective of ``fit_attr`` is
    used. The returned list holds references to the input individuals.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.
        fit_attr: Attribute used as the selection criterion.

    Returns:
        The selected individuals.
    """
    if sel_count <= 0 or not individuals:
        return []

    key = attrgetter(fit_attr)
    sorted_ = sorted(individuals, key=key, reverse=True)
    floor = min(getattr(ind, fit_attr).wvalues[0] for ind in individuals)
    sum_fits = sum(_wheel_weight(getattr(ind, fit_attr), floor) for ind in individuals)
    if sum_fits == 0:
        return [rng.choice(individuals) for _ in range(sel_count)]

    distance = sum_fits / float(sel_count)
    start = rng.uniform(0, distance)
    points = [start + i * distance for i in range(sel_count)]

    chosen = []
    for p in points:
        i = 0
        sum_ = _wheel_weight(getattr(sorted_[i], fit_attr), floor)
        while sum_ < p:
            i += 1
            sum_ += _wheel_weight(getattr(sorted_[i], fit_attr), floor)
        chosen.append(sorted_[i])

    return chosen

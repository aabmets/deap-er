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
from operator import attrgetter

from deap_er.base.dtypes import Individual
from deap_er.rng import rng

__all__ = [
    "sel_random",
    "sel_best",
    "sel_worst",
    "sel_roulette",
    "sel_stochastic_universal_sampling",
]


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

    Each draw uses only the first objective of ``fit_attr``. The
    returned list holds references to the input individuals.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.
        fit_attr: Attribute used as the selection criterion.

    Returns:
        The selected individuals.
    """
    key = attrgetter(fit_attr)
    sorted_ = sorted(individuals, key=key, reverse=True)
    sum_fits = sum(getattr(ind, fit_attr).values[0] for ind in individuals)
    chosen = []
    for _ in range(sel_count):
        u = rng.random() * sum_fits
        sum_ = 0
        for ind in sorted_:
            sum_ += getattr(ind, fit_attr).values[0]
            if sum_ > u:
                chosen.append(ind)
                break

    return chosen


def sel_stochastic_universal_sampling(
    individuals: list[Individual], sel_count: int, fit_attr: str = "fitness"
) -> list[Individual]:
    """Select ``sel_count`` individuals by stochastic universal sampling.

    A single random offset samples the wheel at evenly spaced
    intervals. Only the first objective of ``fit_attr`` is used. The
    returned list holds references to the input individuals.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.
        fit_attr: Attribute used as the selection criterion.

    Returns:
        The selected individuals.
    """
    if sel_count <= 0:
        return []

    key = attrgetter(fit_attr)
    sorted_ = sorted(individuals, key=key, reverse=True)
    sum_fits = sum(getattr(ind, fit_attr).values[0] for ind in individuals)

    distance = sum_fits / float(sel_count)
    start = rng.uniform(0, distance)
    points = [start + i * distance for i in range(sel_count)]

    chosen = []
    for p in points:
        i = 0
        sum_ = getattr(sorted_[i], fit_attr).values[0]
        while sum_ < p:
            i += 1
            sum_ += getattr(sorted_[i], fit_attr).values[0]
        chosen.append(sorted_[i])

    return chosen

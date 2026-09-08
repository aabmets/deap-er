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

import bisect
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


def _wheel_prefix(
    individuals: list[Individual], fit_attr: str
) -> tuple[list[Individual], list[float], float] | None:
    """Build a sorted wheel with prefix sums, or None if the total is 0.

    Args:
        individuals: Non-empty pool to rank.
        fit_attr: Attribute used as the selection criterion.

    Returns:
        Sorted individuals (best first), inclusive prefix sums of the
        sorted wheel, and the encounter-order total used to scale
        draws. ``None`` when every slice is zero.
    """
    sorted_ = sorted(individuals, key=attrgetter(fit_attr), reverse=True)
    floor = min(getattr(ind, fit_attr).wvalues[0] for ind in individuals)
    total = 0.0
    for ind in individuals:
        total += _wheel_weight(getattr(ind, fit_attr), floor)
    if total == 0:
        return None
    prefix: list[float] = []
    acc = 0.0
    for ind in sorted_:
        acc += _wheel_weight(getattr(ind, fit_attr), floor)
        prefix.append(acc)
    return sorted_, prefix, total


def sel_random(individuals: list[Individual], sel_count: int) -> list[Individual]:
    """Select ``sel_count`` individuals uniformly at random.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.

    Returns:
        The selected individuals. An empty pool or ``sel_count <= 0``
        returns an empty list.
    """
    if sel_count <= 0 or not individuals:
        return []
    return [rng.choice(individuals) for _ in range(sel_count)]


def sel_best(
    individuals: list[Individual], sel_count: int, fit_attr: str = "fitness"
) -> list[Individual]:
    """Select the ``sel_count`` best individuals.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select. ``sel_count <= 0``
            returns an empty list.
        fit_attr: Attribute used as the selection criterion.

    Returns:
        The selected individuals.
    """
    if sel_count <= 0:
        return []
    key = attrgetter(fit_attr)
    return sorted(individuals, key=key, reverse=True)[:sel_count]


def sel_worst(
    individuals: list[Individual], sel_count: int, fit_attr: str = "fitness"
) -> list[Individual]:
    """Select the ``sel_count`` worst individuals.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select. ``sel_count <= 0``
            returns an empty list.
        fit_attr: Attribute used as the selection criterion.

    Returns:
        The selected individuals.
    """
    if sel_count <= 0:
        return []
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
    wheel = _wheel_prefix(individuals, fit_attr)
    if wheel is None:
        return [rng.choice(individuals) for _ in range(sel_count)]
    sorted_, prefix, total = wheel
    chosen = []
    for _ in range(sel_count):
        idx = bisect.bisect_right(prefix, rng.random() * total)
        if idx >= len(sorted_):
            idx = len(sorted_) - 1
        chosen.append(sorted_[idx])
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
    wheel = _wheel_prefix(individuals, fit_attr)
    if wheel is None:
        return [rng.choice(individuals) for _ in range(sel_count)]
    sorted_, prefix, total = wheel
    distance = total / float(sel_count)
    start = rng.uniform(0, distance)
    chosen = []
    for i in range(sel_count):
        idx = bisect.bisect_left(prefix, start + i * distance)
        if idx >= len(sorted_):
            idx = len(sorted_) - 1
        chosen.append(sorted_[idx])
    return chosen

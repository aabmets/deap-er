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
    from deap_er.private.typedefs import Individual, Mates, NumOrSeq
from deap_er.private.various.rng import rng

from .bounds import broadcast_param, require_positive_eta
from .cx_point import slicer

__all__: list[str] = [
    "cx_blend",
    "cx_blend_bounded",
    "cx_es_blend",
    "cx_simulated_binary",
    "cx_simulated_binary_bounded",
    "cx_uniform",
]


def cx_blend(ind1: Individual, ind2: Individual, alpha: float) -> Mates:
    """Execute a blend crossover on two individuals.

    Both individuals are modified in place.

    Args:
        ind1: The first individual.
        ind2: The second individual.
        alpha: Extent of the interval in which the new values can be
            drawn for each attribute on both sides of the parents'
            attributes.

    Returns:
        The two individuals after crossover.
    """
    for i, (x1, x2) in enumerate(zip(ind1, ind2, strict=False)):
        gamma = (1.0 + 2.0 * alpha) * rng.random() - alpha
        ind1[i] = (1.0 - gamma) * x1 + gamma * x2
        ind2[i] = gamma * x1 + (1.0 - gamma) * x2

    return ind1, ind2


def cx_blend_bounded(
    ind1: Individual, ind2: Individual, alpha: float, low: NumOrSeq, up: NumOrSeq
) -> Mates:
    """Execute a bounded blend crossover on two individuals.

    Both individuals are modified in place. Each child gene is
    clamped to ``[low, up]`` after the blend draw.

    Args:
        ind1: The first individual.
        ind2: The second individual.
        alpha: Extent of the interval in which the new values can be
            drawn for each attribute on both sides of the parents'
            attributes.
        low: Lower bound of the search space.
        up: Upper bound of the search space.

    Returns:
        The two individuals after crossover.

    Raises:
        ValueError: If a bound sequence is shorter than the shorter
            individual.
    """
    size = min(len(ind1), len(ind2))
    low = broadcast_param("low", low, size, "the shorter individual")
    up = broadcast_param("up", up, size, "the shorter individual")

    for i, xl, xu in zip(list(range(size)), low, up, strict=False):
        if xu <= xl:
            continue
        x1, x2 = ind1[i], ind2[i]
        gamma = (1.0 + 2.0 * alpha) * rng.random() - alpha
        ind1[i] = min(max((1.0 - gamma) * x1 + gamma * x2, xl), xu)
        ind2[i] = min(max(gamma * x1 + (1.0 - gamma) * x2, xl), xu)

    return ind1, ind2


def cx_es_blend(ind1: Individual, ind2: Individual, alpha: float) -> Mates:
    """Execute a blend crossover on two individuals and their strategies.

    Both individuals and their ``strategy`` vectors are modified in place.

    Args:
        ind1: The first individual.
        ind2: The second individual.
        alpha: Extent of the interval in which the new values can be
            drawn for each attribute on both sides of the parents'
            attributes.

    Returns:
        The two individuals after crossover.
    """
    zipper = zip(ind1, ind1.strategy, ind2, ind2.strategy, strict=False)
    for i, (x1, s1, x2, s2) in enumerate(zipper):
        gamma = (1.0 + 2.0 * alpha) * rng.random() - alpha
        ind1[i] = (1.0 - gamma) * x1 + gamma * x2
        ind2[i] = gamma * x1 + (1.0 - gamma) * x2

        gamma = (1.0 + 2.0 * alpha) * rng.random() - alpha
        ind1.strategy[i] = (1.0 - gamma) * s1 + gamma * s2
        ind2.strategy[i] = gamma * s1 + (1.0 - gamma) * s2

    return ind1, ind2


def cx_simulated_binary(ind1: Individual, ind2: Individual, eta: float) -> Mates:
    """Execute a simulated binary crossover on two individuals.

    Both individuals are modified in place.

    Args:
        ind1: The first individual.
        ind2: The second individual.
        eta: Crowding degree of the crossover. Higher values produce
            children more similar to their parents; smaller values
            produce children more divergent from their parents.

    Returns:
        The two individuals after crossover.
    """
    for i, (x1, x2) in enumerate(zip(ind1, ind2, strict=False)):
        rand = rng.random()

        beta = 2.0 * rand if rand <= 0.5 else 1.0 / (2.0 * (1.0 - rand))

        beta **= 1.0 / (eta + 1.0)
        ind1[i] = 0.5 * (((1 + beta) * x1) + ((1 - beta) * x2))
        ind2[i] = 0.5 * (((1 - beta) * x1) + ((1 + beta) * x2))

    return ind1, ind2


def cx_simulated_binary_bounded(
    ind1: Individual, ind2: Individual, eta: float, low: NumOrSeq, up: NumOrSeq
) -> Mates:
    """Execute a bounded simulated binary crossover on two individuals.

    Both individuals are modified in place.

    Args:
        ind1: The first individual.
        ind2: The second individual.
        eta: Crowding degree of the crossover. Higher values produce
            children more similar to their parents; smaller values
            produce children more divergent from their parents.
        low: Lower bound of the search space.
        up: Upper bound of the search space.

    Returns:
        The two individuals after crossover.

    Raises:
        ValueError: If ``eta`` is not greater than 0, or if a bound
            sequence is shorter than the shorter individual.
    """
    require_positive_eta(eta)

    def calc_c(diff: float, side: float) -> float:
        """Map a gap to the bound into one bounded SBX child value.

        Args:
            diff: Distance from the nearer parent to the active bound.
            side: ``-1`` for the lower child, ``+1`` for the upper child.

        Returns:
            One child coordinate for the current parent pair.
        """
        beta = 1.0 + (2.0 * diff / (x2 - x1))
        alpha = 2.0 - beta ** -(eta + 1)
        if rand <= 1.0 / alpha:
            beta_q = (rand * alpha) ** (1.0 / (eta + 1))
        else:
            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
        c = 0.5 * (x1 + x2 + side * beta_q * (x2 - x1))
        return float(c)

    size = min(len(ind1), len(ind2))
    low = broadcast_param("low", low, size, "the shorter individual")
    up = broadcast_param("up", up, size, "the shorter individual")

    for i, xl, xu in zip(list(range(size)), low, up, strict=False):
        if xu <= xl:
            continue
        if rng.random() <= 0.5 and abs(ind1[i] - ind2[i]) > 1e-14:
            x1 = min(max(min(ind1[i], ind2[i]), xl), xu)
            x2 = min(max(max(ind1[i], ind2[i]), xl), xu)
            if abs(x1 - x2) <= 1e-14:
                continue
            rand = rng.random()

            c1 = calc_c(x1 - xl, -1.0)
            c1 = min(max(c1, xl), xu)

            c2 = calc_c(xu - x2, 1.0)
            c2 = min(max(c2, xl), xu)

            if rng.random() <= 0.5:
                ind1[i] = c2
                ind2[i] = c1
            else:
                ind1[i] = c1
                ind2[i] = c2

    return ind1, ind2


def cx_uniform(ind1: Individual, ind2: Individual, cx_prob: float) -> Mates:
    """Execute a uniform crossover on two individuals.

    Both individuals are modified in place.

    Args:
        ind1: The first individual.
        ind2: The second individual.
        cx_prob: Probability of swapping any two traits.

    Returns:
        The two individuals after crossover.
    """
    size = min(len(ind1), len(ind2))
    for i in range(size):
        if rng.random() < cx_prob:
            slicer(ind1, ind2, i, i + 1)
    return ind1, ind2

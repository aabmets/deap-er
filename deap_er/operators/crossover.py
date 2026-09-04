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

from deap_er.base.dtypes import Individual, Mates, NumOrSeq
from deap_er.rng import rng

from ._bounds import _broadcast_param

__all__ = [
    "cx_one_point",
    "cx_messy_one_point",
    "cx_two_point",
    "cx_two_point_copy",
    "cx_es_two_point",
    "cx_es_two_point_copy",
    "cx_partially_matched",
    "cx_uniform_partially_matched",
    "cx_blend",
    "cx_es_blend",
    "cx_simulated_binary",
    "cx_simulated_binary_bounded",
    "cx_uniform",
    "cx_ordered",
]


def _slicer(
    ind1: Individual, ind2: Individual, start: int, stop: int | None = None, copy: bool = False
) -> Mates:
    """Swap a segment of two individuals.

    Both individuals are modified in place. When ``stop`` is omitted,
    each individual is sliced from ``start`` to its own length.

    Args:
        ind1: The first individual.
        ind2: The second individual.
        start: First index of the exchanged segment.
        stop: Exclusive end index of the exchanged segment. If omitted,
            each individual is sliced to its own length.
        copy: If True, copy the slices before assignment. Use this for
            individuals backed by numpy arrays.

    Returns:
        The two individuals after the swap.
    """
    if stop is None:
        s1 = slice(start, len(ind1))
        s2 = slice(start, len(ind2))
    else:
        s1 = slice(start, stop)
        s2 = slice(start, stop)

    temp_1 = ind1[s1] if not copy else ind1[s1].copy()
    temp_2 = ind2[s2] if not copy else ind2[s2].copy()
    ind1[s1] = temp_2
    ind2[s2] = temp_1
    return ind1, ind2


def _two_point(
    ind1: Individual, ind2: Individual, copy: bool = False, strategy: bool = False
) -> Mates:
    """Execute a two-point crossover on two individuals.

    Both individuals are modified in place.

    Args:
        ind1: The first individual.
        ind2: The second individual.
        copy: If True, copy the exchanged slices before assignment.
        strategy: If True, apply the same cut points to each
            individual's ``strategy`` vector.

    Returns:
        The two individuals after crossover.
    """
    size = min(len(ind1), len(ind2))
    cxp1 = rng.randint(1, size)
    cxp2 = rng.randint(1, size - 1)
    if cxp2 >= cxp1:
        cxp2 += 1
    else:
        cxp1, cxp2 = cxp2, cxp1
    ind1, ind2 = _slicer(ind1, ind2, cxp1, cxp2, copy)
    if strategy:
        _slicer(ind1.strategy, ind2.strategy, cxp1, cxp2)
    return ind1, ind2


def _match(ind1: Individual, ind2: Individual, p1: list[int], p2: list[int], i: int) -> None:
    """Swap the alleles at one locus and keep the PMX position maps valid.

    Both individuals and both maps are modified in place.

    Args:
        ind1: The first individual.
        ind2: The second individual.
        p1: Allele-to-index map for ``ind1``.
        p2: Allele-to-index map for ``ind2``.
        i: Locus at which to exchange alleles.
    """
    temp1, temp2 = ind1[i], ind2[i]
    ind1[i], ind1[p1[temp2]] = temp2, temp1
    ind2[i], ind2[p2[temp1]] = temp1, temp2
    p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
    p2[temp1], p2[temp2] = p2[temp2], p2[temp1]


def cx_one_point(ind1: Individual, ind2: Individual) -> Mates:
    """Execute a one-point crossover on two individuals.

    Both individuals are modified in place.

    Args:
        ind1: The first individual.
        ind2: The second individual.

    Returns:
        The two individuals after crossover.
    """
    size = min(len(ind1), len(ind2))
    cxp = rng.randint(1, size - 1)
    ind1, ind2 = _slicer(ind1, ind2, cxp)
    return ind1, ind2


def cx_messy_one_point(ind1: Individual, ind2: Individual) -> Mates:
    """Execute a messy one-point crossover on two individuals.

    Cut points are chosen independently, so the individuals may change
    length. Both individuals are modified in place.

    Args:
        ind1: The first individual.
        ind2: The second individual.

    Returns:
        The two individuals after crossover.
    """
    cxp1 = rng.randint(0, len(ind1))
    cxp2 = rng.randint(0, len(ind2))
    ind1, ind2 = _slicer(ind1, ind2, cxp1, cxp2)
    return ind1, ind2


def cx_two_point(ind1: Individual, ind2: Individual) -> Mates:
    """Execute a two-point crossover on two individuals.

    Both individuals are modified in place.

    Args:
        ind1: The first individual.
        ind2: The second individual.

    Returns:
        The two individuals after crossover.
    """
    ind1, ind2 = _two_point(ind1, ind2)
    return ind1, ind2


def cx_two_point_copy(ind1: Individual, ind2: Individual) -> Mates:
    """Execute a two-point crossover on copies of two individuals.

    Use this instead of ``cx_two_point`` when the individuals are
    based on numpy arrays, to avoid incorrect mating behavior due
    to the specifics of the numpy array datatype.

    Args:
        ind1: The first individual.
        ind2: The second individual.

    Returns:
        The two individuals after crossover.
    """
    ind1, ind2 = _two_point(ind1, ind2, copy=True)
    return ind1, ind2


def cx_es_two_point(ind1: Individual, ind2: Individual) -> Mates:
    """Execute a two-point crossover on two individuals and their strategies.

    Both individuals and their ``strategy`` vectors are modified in place.

    Args:
        ind1: The first individual.
        ind2: The second individual.

    Returns:
        The two individuals after crossover.
    """
    ind1, ind2 = _two_point(ind1, ind2, strategy=True)
    return ind1, ind2


def cx_es_two_point_copy(ind1: Individual, ind2: Individual) -> Mates:
    """Execute a two-point crossover on copies of two individuals and their strategies.

    Use this instead of ``cx_es_two_point`` when the individuals are
    based on numpy arrays, to avoid incorrect mating behavior due to
    the specifics of the numpy array datatype.

    Args:
        ind1: The first individual.
        ind2: The second individual.

    Returns:
        The two individuals after crossover.
    """
    ind1, ind2 = _two_point(ind1, ind2, copy=True, strategy=True)
    return ind1, ind2


def cx_partially_matched(ind1: Individual, ind2: Individual) -> Mates:
    """Execute a partially matched crossover on two individuals.

    Both individuals are modified in place.

    Args:
        ind1: The first individual.
        ind2: The second individual.

    Returns:
        The two individuals after crossover.
    """
    size = min(len(ind1), len(ind2))
    p1, p2 = [0] * size, [0] * size

    cxp1 = rng.randint(0, size)
    cxp2 = rng.randint(0, size - 1)

    if cxp2 >= cxp1:
        cxp2 += 1
    else:
        cxp1, cxp2 = cxp2, cxp1

    for i in range(size):
        p1[ind1[i]] = i
        p2[ind2[i]] = i

    for i in range(cxp1, cxp2):
        _match(ind1, ind2, p1, p2, i)

    return ind1, ind2


def cx_uniform_partially_matched(ind1: Individual, ind2: Individual, cx_prob: float) -> Mates:
    """Execute a uniform partially matched crossover on two individuals.

    Both individuals are modified in place.

    Args:
        ind1: The first individual.
        ind2: The second individual.
        cx_prob: Probability of swapping any two traits.

    Returns:
        The two individuals after crossover.
    """
    size = min(len(ind1), len(ind2))
    p1, p2 = [0] * size, [0] * size

    for i in range(size):
        p1[ind1[i]] = i
        p2[ind2[i]] = i

    for i in range(size):
        if rng.random() < cx_prob:
            _match(ind1, ind2, p1, p2, i)

    return ind1, ind2


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
        ValueError: If a bound sequence is shorter than the shorter
            individual.
    """

    def calc_c(diff: float) -> float:
        """Map a gap to the bound into one bounded SBX child value.

        Args:
            diff: Distance from the nearer parent to the active bound.

        Returns:
            One child coordinate for the current parent pair.
        """
        beta = 1.0 + (2.0 * diff / (x2 - x1))
        alpha = 2.0 - beta ** -(eta + 1)
        if rand <= 1.0 / alpha:
            beta_q = (rand * alpha) ** (1.0 / (eta + 1))
        else:
            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
        c = 0.5 * (x1 + x2 - beta_q * (x2 - x1))
        return float(c)

    size = min(len(ind1), len(ind2))
    low = _broadcast_param("low", low, size, "the shorter individual")
    up = _broadcast_param("up", up, size, "the shorter individual")

    for i, xl, xu in zip(list(range(size)), low, up, strict=False):
        if rng.random() <= 0.5 and abs(ind1[i] - ind2[i]) > 1e-14:
            x1 = min(ind1[i], ind2[i])
            x2 = max(ind1[i], ind2[i])
            rand = rng.random()

            c1 = calc_c(x1 - xl)
            c1 = min(max(c1, xl), xu)

            c2 = calc_c(xu - x2)
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
            _slicer(ind1, ind2, i, i + 1)
    return ind1, ind2


def cx_ordered(ind1: Individual, ind2: Individual) -> Mates:
    """Execute an ordered crossover on two individuals.

    Both individuals are modified in place.

    Args:
        ind1: The first individual.
        ind2: The second individual.

    Returns:
        The two individuals after crossover.
    """
    size = min(len(ind1), len(ind2))
    a, b = rng.sample(list(range(size)), 2)
    if a > b:
        a, b = b, a

    holes1, holes2 = [True] * size, [True] * size
    for i in range(size):
        if i < a or i > b:
            holes1[ind2[i]] = False
            holes2[ind1[i]] = False

    temp1, temp2 = ind1, ind2
    k1, k2 = b + 1, b + 1

    for i in range(size):
        if not holes1[temp1[(i + b + 1) % size]]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not holes2[temp2[(i + b + 1) % size]]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1

    for i in range(a, b + 1):
        _slicer(ind1, ind2, i, i + 1)

    return ind1, ind2

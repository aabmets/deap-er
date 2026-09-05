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
from deap_er.base.typedefs import Individual, Mates
from deap_er.rng import rng

from ._cx_point import _slicer

__all__: list[str] = []


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

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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual, Mates

from deap_er.private.various.rng import rng

from .cx_point import slicer

__all__: list[str] = ["match", "cx_partially_matched", "cx_uniform_partially_matched", "cx_ordered"]

_PERM_DUP = (
    "Partially matched and ordered crossover require individuals "
    "to be permutations without duplicate alleles."
)
_PERM_SET = (
    "Partially matched and ordered crossover require both "
    "individuals to contain the same set of alleles."
)


def _allele_maps(
    ind1: Individual, ind2: Individual, size: int
) -> tuple[dict[Any, int], dict[Any, int]]:
    """Build allele-to-index maps and require a shared permutation.

    Args:
        ind1: The first individual.
        ind2: The second individual.
        size: Number of leading genes to map.

    Returns:
        Allele-to-index maps for ``ind1`` and ``ind2``.

    Raises:
        ValueError: If either individual has duplicate alleles or the
            two gene sets differ.
    """
    p1: dict[Any, int] = {}
    p2: dict[Any, int] = {}
    for i in range(size):
        a1, a2 = ind1[i], ind2[i]
        if a1 in p1 or a2 in p2:
            raise ValueError(_PERM_DUP)
        p1[a1] = i
        p2[a2] = i
    if p1.keys() != p2.keys():
        raise ValueError(_PERM_SET)
    return p1, p2


def match(
    ind1: Individual, ind2: Individual, p1: dict[Any, int], p2: dict[Any, int], i: int
) -> None:
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

    Both individuals are modified in place. Alleles may be any
    hashable values that form a shared permutation.

    Args:
        ind1: The first individual.
        ind2: The second individual.

    Returns:
        The two individuals after crossover.

    Raises:
        ValueError: If the individuals are not permutations of the
            same allele set.
    """
    size = min(len(ind1), len(ind2))
    p1, p2 = _allele_maps(ind1, ind2, size)
    if size < 2:
        return ind1, ind2

    cxp1 = rng.randint(0, size)
    cxp2 = rng.randint(0, size - 1)

    if cxp2 >= cxp1:
        cxp2 += 1
    else:
        cxp1, cxp2 = cxp2, cxp1

    for i in range(cxp1, cxp2):
        match(ind1, ind2, p1, p2, i)

    return ind1, ind2


def cx_uniform_partially_matched(ind1: Individual, ind2: Individual, cx_prob: float) -> Mates:
    """Execute a uniform partially matched crossover on two individuals.

    Both individuals are modified in place. Alleles may be any
    hashable values that form a shared permutation.

    Args:
        ind1: The first individual.
        ind2: The second individual.
        cx_prob: Probability of swapping any two traits.

    Returns:
        The two individuals after crossover.

    Raises:
        ValueError: If the individuals are not permutations of the
            same allele set.
    """
    size = min(len(ind1), len(ind2))
    p1, p2 = _allele_maps(ind1, ind2, size)

    for i in range(size):
        if rng.random() < cx_prob:
            match(ind1, ind2, p1, p2, i)

    return ind1, ind2


def cx_ordered(ind1: Individual, ind2: Individual) -> Mates:
    """Execute an ordered crossover on two individuals.

    Both individuals are modified in place. Alleles may be any
    hashable values that form a shared permutation.

    Args:
        ind1: The first individual.
        ind2: The second individual.

    Returns:
        The two individuals after crossover.

    Raises:
        ValueError: If the individuals are not permutations of the
            same allele set.
    """
    size = min(len(ind1), len(ind2))
    _allele_maps(ind1, ind2, size)
    if size < 2:
        return ind1, ind2
    a, b = rng.sample(list(range(size)), 2)
    if a > b:
        a, b = b, a

    holes1 = {ind2[i] for i in range(a, b + 1)}
    holes2 = {ind1[i] for i in range(a, b + 1)}

    temp1, temp2 = ind1, ind2
    k1, k2 = b + 1, b + 1

    for i in range(size):
        src1 = temp1[(i + b + 1) % size]
        if src1 not in holes1:
            ind1[k1 % size] = src1
            k1 += 1

        src2 = temp2[(i + b + 1) % size]
        if src2 not in holes2:
            ind2[k2 % size] = src2
            k2 += 1

    for i in range(a, b + 1):
        slicer(ind1, ind2, i, i + 1)

    return ind1, ind2

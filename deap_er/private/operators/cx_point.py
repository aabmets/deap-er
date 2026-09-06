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
    from deap_er.private.typedefs import Individual, Mates
from deap_er.private.various.rng import rng

__all__: list[str] = [
    "slicer",
    "two_point",
    "cx_one_point",
    "cx_messy_one_point",
    "cx_two_point",
    "cx_two_point_copy",
    "cx_es_two_point",
    "cx_es_two_point_copy",
]


def slicer(
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

    temp_1 = ind1[s1].copy() if copy or hasattr(ind1[s1], "copy") else list(ind1[s1])
    temp_2 = ind2[s2].copy() if copy or hasattr(ind2[s2], "copy") else list(ind2[s2])
    ind1[s1] = temp_2
    ind2[s2] = temp_1
    return ind1, ind2


def two_point(
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
    if size < 2:
        return ind1, ind2
    cxp1 = rng.randint(1, size)
    cxp2 = rng.randint(1, size - 1)
    if cxp2 >= cxp1:
        cxp2 += 1
    else:
        cxp1, cxp2 = cxp2, cxp1
    ind1, ind2 = slicer(ind1, ind2, cxp1, cxp2, copy)
    if strategy:
        slicer(ind1.strategy, ind2.strategy, cxp1, cxp2, copy)
    return ind1, ind2


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
    if size < 2:
        return ind1, ind2
    cxp = rng.randint(1, size - 1)
    ind1, ind2 = slicer(ind1, ind2, cxp)
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
    ind1[cxp1:], ind2[cxp2:] = list(ind2[cxp2:]), list(ind1[cxp1:])
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
    ind1, ind2 = two_point(ind1, ind2)
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
    ind1, ind2 = two_point(ind1, ind2, copy=True)
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
    ind1, ind2 = two_point(ind1, ind2, strategy=True)
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
    ind1, ind2 = two_point(ind1, ind2, copy=True, strategy=True)
    return ind1, ind2

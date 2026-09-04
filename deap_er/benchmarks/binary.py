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
from deap_er.base.dtypes import *

__all__ = ["bm_royal_road_1", "bm_royal_road_2", "bm_chuang_f1", "bm_chuang_f2", "bm_chuang_f3"]


def bm_royal_road_1(individual: Individual, order: int) -> tuple[int]:
    """Evaluate Royal Road function R1.

    As presented by Melanie Mitchell in "An introduction to Genetic
    Algorithms".

    Args:
        individual: Individual to evaluate.
        order: Order of the royal road function.

    Returns:
        The royal road function value.
    """
    nelem = len(individual) // order
    max_value = int(2**order - 1)
    total = 0
    for i in range(nelem):
        start = i * order
        stop = i * order + order
        values = individual[start:stop]
        mapper = map(str, values)
        gene = int("".join(mapper), 2)
        total += order * int(gene / max_value)
    return (total,)


def bm_royal_road_2(individual: Individual, order: int) -> tuple[int]:
    """Evaluate Royal Road function R2.

    As presented by Melanie Mitchell in "An introduction to Genetic
    Algorithms".

    Args:
        individual: Individual to evaluate.
        order: Order of the royal road function.

    Returns:
        The royal road function value.
    """
    total = 0
    n_order = order
    while n_order < order**2:
        total += bm_royal_road_1(individual, n_order)[0]
        n_order *= 2
    return (total,)


def bm_chuang_f1(individual: Individual) -> tuple[int]:
    """Evaluate Chuang and Hsu's first binary deceptive function.

    From "Multivariate Multi-Model Approach for Globally Multimodal
    Problems". Two global optima at all-ones and all-zeros. The
    individual must have 41 dimensions.

    Args:
        individual: Individual to evaluate.

    Returns:
        The deceptive function value.
    """
    total = 0
    if individual[-1] == 0:
        for i in range(0, len(individual) - 1, 4):
            total += _inv_trap(individual[i : i + 4])
    else:
        for i in range(0, len(individual) - 1, 4):
            total += _trap(individual[i : i + 4])
    return (total,)


def bm_chuang_f2(individual: Individual) -> tuple[int]:
    """Evaluate Chuang and Hsu's second binary deceptive function.

    From "Multivariate Multi-Model Approach for Globally Multimodal
    Problems". Four global optima: half-and-half, reverse half-and-half,
    all-ones, and all-zeros. The individual must have 41 dimensions.

    Args:
        individual: Individual to evaluate.

    Returns:
        The deceptive function value.
    """
    total = 0
    if individual[-2] == 0 and individual[-1] == 0:
        for i in range(0, len(individual) - 2, 8):
            total += _inv_trap(individual[i : i + 4]) + _inv_trap(individual[i + 4 : i + 8])
    elif individual[-2] == 0 and individual[-1] == 1:
        for i in range(0, len(individual) - 2, 8):
            total += _inv_trap(individual[i : i + 4]) + _trap(individual[i + 4 : i + 8])
    elif individual[-2] == 1 and individual[-1] == 0:
        for i in range(0, len(individual) - 2, 8):
            total += _trap(individual[i : i + 4]) + _inv_trap(individual[i + 4 : i + 8])
    else:
        for i in range(0, len(individual) - 2, 8):
            total += _trap(individual[i : i + 4]) + _trap(individual[i + 4 : i + 8])
    return (total,)


def bm_chuang_f3(individual: Individual) -> tuple[int]:
    """Evaluate Chuang and Hsu's third binary deceptive function.

    From "Multivariate Multi-Model Approach for Globally Multimodal
    Problems". Two global optima at all-ones and all-zeros. The
    individual must have 41 dimensions.

    Args:
        individual: Individual to evaluate.

    Returns:
        The deceptive function value.
    """
    total = 0
    if individual[-1] == 0:
        for i in range(0, len(individual) - 1, 4):
            total += _inv_trap(individual[i : i + 4])
    else:
        for i in range(2, len(individual) - 3, 4):
            total += _inv_trap(individual[i : i + 4])
        total += _trap(individual[-2:] + individual[:2])
    return (total,)


def _trap(individual: Individual) -> int:
    """Score a binary block with a deceptive all-ones trap.

    Args:
        individual: Binary block to score.

    Returns:
        The trap value for the block.
    """
    u = sum(individual)
    k = len(individual)
    return int(k if u == k else k - 1 - u)


def _inv_trap(individual: Individual) -> int:
    """Score a binary block with a deceptive all-zeros trap.

    Args:
        individual: Binary block to score.

    Returns:
        The inverse-trap value for the block.
    """
    u = sum(individual)
    k = len(individual)
    return int(k if u == 0 else u - 1)

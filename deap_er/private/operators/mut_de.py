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
    from deap_er.private.typedefs import Individual, Mutant, NumOrSeq
from deap_er.private.various.rng import rng

from .bounds import broadcast_param

__all__: list[str] = ["mut_de"]

_BOUNDS_PAIR = "Arguments 'low' and 'up' must both be set or both omitted."
_DONOR_SHORT = "Donors a, b, and c must be at least the size of the individual"


def mut_de(
    individual: Individual,
    a: Individual,
    b: Individual,
    c: Individual,
    scale: float,
    cx_prob: float,
    *,
    low: NumOrSeq | None = None,
    up: NumOrSeq | None = None,
) -> Mutant:
    """Write a DE/rand/1/bin trial onto *individual*.

    Genes selected by the binomial mask (rate ``cx_prob``, at least
    one gene forced) become ``a[i] + scale * (b[i] - c[i])``. The
    individual is modified in place. The caller supplies donors and
    keeps the trial when it is better. Optional ``low`` / ``up``
    clamp written genes only.

    Args:
        individual: Trial vector to overwrite. Clone the parent first.
        a: Base donor.
        b: First difference donor.
        c: Second difference donor.
        scale: Difference weight ``F``.
        cx_prob: Per-gene crossover rate ``CR``.
        low: Lower bound of the search space. Optional.
        up: Upper bound of the search space. Optional.

    Returns:
        A one-element tuple containing the trial individual.

    Raises:
        ValueError: If a donor is shorter than the individual, if
            only one of ``low`` / ``up`` is set, or if a bound
            sequence is shorter than the individual.
    """
    size = len(individual)
    if size == 0:
        return (individual,)
    if min(len(a), len(b), len(c)) < size:
        raise ValueError(f"{_DONOR_SHORT}: {min(len(a), len(b), len(c))} < {size}")
    if (low is None) ^ (up is None):
        raise ValueError(_BOUNDS_PAIR)

    lows = ups = None
    if low is not None and up is not None:
        lows = broadcast_param("low", low, size)
        ups = broadcast_param("up", up, size)

    index = rng.randrange(size)
    draws = rng.take_floats(size)
    for i, draw in enumerate(draws):
        if i != index and draw >= cx_prob:
            continue
        gene = a[i] + scale * (b[i] - c[i])
        if lows is not None and ups is not None:
            xl, xu = lows[i], ups[i]
            if xu > xl:
                gene = min(max(gene, xl), xu)
        individual[i] = gene

    return (individual,)

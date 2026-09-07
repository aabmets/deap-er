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

__all__: list[str] = ["mut_gaussian_bounded"]


def mut_gaussian_bounded(
    individual: Individual,
    mu: NumOrSeq,
    sigma: NumOrSeq,
    low: NumOrSeq,
    up: NumOrSeq,
    mut_prob: float,
) -> Mutant:
    """Apply a Gaussian mutation and clamp each mutated gene into a box.

    The individual is modified in place. The draw is the same
    ``N(mu, sigma)`` add as ``mut_gaussian``. ``mu``, ``sigma``,
    ``low``, and ``up`` may be scalars or per-gene sequences. An empty
    interval (``up <= low``) is skipped.

    Args:
        individual: Individual to mutate.
        mu: Mean of the Gaussian mutation.
        sigma: Standard deviation of the Gaussian mutation.
        low: Lower bound of the search space.
        up: Upper bound of the search space.
        mut_prob: Probability of mutating each attribute.

    Returns:
        A one-element tuple containing the mutated individual.

    Raises:
        ValueError: If ``mu``, ``sigma``, ``low``, or ``up`` is a
            sequence shorter than the individual.
    """
    size = len(individual)
    mu = broadcast_param("mu", mu, size)
    sigma = broadcast_param("sigma", sigma, size)
    low = broadcast_param("low", low, size)
    up = broadcast_param("up", up, size)

    idx = list(range(size))
    for i, m, s, xl, xu in zip(idx, mu, sigma, low, up, strict=False):
        if rng.random() < mut_prob:
            if xu <= xl:
                continue
            individual[i] = min(max(individual[i] + rng.gauss(m, s), xl), xu)

    return (individual,)

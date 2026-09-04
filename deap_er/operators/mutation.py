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
import math

from deap_er.base.dtypes import *
from deap_er.rng import rng

from ._bounds import _broadcast_param

__all__ = [
    "mut_gaussian",
    "mut_polynomial_bounded",
    "mut_shuffle_indexes",
    "mut_flip_bit",
    "mut_uniform_int",
    "mut_es_log_normal",
]


def mut_gaussian(individual: Individual, mu: NumOrSeq, sigma: NumOrSeq, mut_prob: float) -> Mutant:
    """Apply a Gaussian mutation of mean *mu* and standard deviation *sigma*.

    The individual is modified in place. ``mu`` and ``sigma`` may be
    scalars or per-gene sequences.

    Args:
        individual: Individual to mutate.
        mu: Mean of the Gaussian mutation.
        sigma: Standard deviation of the Gaussian mutation.
        mut_prob: Probability of mutating each attribute.

    Returns:
        A one-element tuple containing the mutated individual.

    Raises:
        ValueError: If ``mu`` or ``sigma`` is a sequence shorter than
            the individual.
    """
    size = len(individual)
    mu = _broadcast_param("mu", mu, size)
    sigma = _broadcast_param("sigma", sigma, size)

    idx = list(range(size))
    for i, m, s in zip(idx, mu, sigma, strict=False):
        if rng.random() < mut_prob:
            individual[i] += rng.gauss(m, s)

    return (individual,)


def mut_polynomial_bounded(
    individual: Individual, eta: float, low: NumOrSeq, up: NumOrSeq, mut_prob: float
) -> Mutant:
    """Apply a bounded polynomial mutation with crowding degree *eta*.

    The individual is modified in place. ``low`` and ``up`` may be
    scalars or per-gene sequences.

    Args:
        individual: Individual to mutate.
        eta: Crowding degree of the mutation. Higher values produce
            children more similar to their parents; smaller values
            produce children more divergent from their parents.
        low: Lower bound of the search space.
        up: Upper bound of the search space.
        mut_prob: Probability of mutating each attribute.

    Returns:
        A one-element tuple containing the mutated individual.

    Raises:
        ValueError: If ``low`` or ``up`` is a sequence shorter than
            the individual.
    """
    size = len(individual)
    low = _broadcast_param("low", low, size)
    up = _broadcast_param("up", up, size)

    idx = list(range(size))
    for i, xl, xu in zip(idx, low, up, strict=False):
        if rng.random() <= mut_prob:
            x = individual[i]
            delta_1 = (x - xl) / (xu - xl)
            delta_2 = (xu - x) / (xu - xl)
            rand = rng.random()
            mut_pow = 1.0 / (eta + 1.0)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                delta_q = val**mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                delta_q = 1.0 - val**mut_pow

            x = x + delta_q * (xu - xl)
            x = min(max(x, xl), xu)
            individual[i] = x

    return (individual,)


def mut_shuffle_indexes(individual: Individual, mut_prob: float) -> Mutant:
    """Shuffle attributes of the individual.

    The individual is modified in place.

    Args:
        individual: Individual to mutate.
        mut_prob: Probability of mutating each attribute.

    Returns:
        A one-element tuple containing the mutated individual.
    """
    size = len(individual)
    for i in range(size):
        if rng.random() < mut_prob:
            swap_indx = rng.randint(0, size - 2)
            if swap_indx >= i:
                swap_indx += 1
            individual[i], individual[swap_indx] = individual[swap_indx], individual[i]

    return (individual,)


def mut_flip_bit(individual: Individual, mut_prob: float) -> Mutant:
    """Flip the values of random attributes of the individual.

    The individual is modified in place.

    Args:
        individual: Individual to mutate.
        mut_prob: Probability of mutating each attribute.

    Returns:
        A one-element tuple containing the mutated individual.
    """
    for i in range(len(individual)):
        if rng.random() < mut_prob:
            individual[i] = type(individual[i])(not individual[i])

    return (individual,)


def mut_uniform_int(individual: Individual, low: int, up: int, mut_prob: float) -> Mutant:
    """Replace attributes with integers drawn uniformly from [*low*, *up*].

    The individual is modified in place. Bounds are inclusive.

    Args:
        individual: Individual to mutate.
        low: Lower bound of the search space.
        up: Upper bound of the search space.
        mut_prob: Probability of mutating each attribute.

    Returns:
        A one-element tuple containing the mutated individual.

    Raises:
        ValueError: If ``low`` or ``up`` is a sequence shorter than
            the individual.
    """
    size = len(individual)
    lows = _broadcast_param("low", low, size)
    ups = _broadcast_param("up", up, size)

    idx = list(range(size))
    for i, xl, xu in zip(idx, lows, ups, strict=False):
        if rng.random() < mut_prob:
            individual[i] = rng.randint(int(xl), int(xu))

    return (individual,)


def mut_es_log_normal(individual: Individual, learn_rate: float, mut_prob: float) -> Mutant:
    """Mutate an evolution strategy according to its ``strategy`` attribute.

    The individual is modified in place. Genes are updated only when
    the individual has a ``strategy`` attribute.

    Args:
        individual: Individual to mutate.
        learn_rate: Learning rate of the evolution strategy. For an
            evolution strategy of (10, 100) the recommended value is 1.
        mut_prob: Probability of mutating each attribute.

    Returns:
        A one-element tuple containing the mutated individual.
    """
    size = len(individual)
    t = learn_rate / math.sqrt(2.0 * math.sqrt(size))
    t0 = learn_rate / math.sqrt(2.0 * size)
    n = rng.gauss(0, 1)
    t0_n = t0 * n

    for indx in range(size):
        if rng.random() < mut_prob and hasattr(individual, "strategy"):
            individual.strategy[indx] *= math.exp(t0_n + t * rng.gauss(0, 1))
            individual[indx] += individual.strategy[indx] * rng.gauss(0, 1)

    return (individual,)

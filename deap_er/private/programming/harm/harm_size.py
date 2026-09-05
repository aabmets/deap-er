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

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deap_er.private.typedefs import GPIndividual
from deap_er.private.various.rng import rng

__all__: list[str] = [
    "target_prob",
    "natural_histogram",
    "cutoff_size",
    "target_histogram",
    "acceptance",
]


def target_prob(
    size: int, alpha: float, beta: float, gamma: float, pop_len: int, decay_start_size: int
) -> float:
    """Return the target acceptance probability for a given tree size.

    The probability decays exponentially past ``decay_start_size``,
    with a half-life that grows linearly with the cutoff.

    Args:
        size: Tree size to score.
        alpha: Half-life scaling factor.
        beta: Minimum half-life.
        gamma: Fraction of individuals allowed past the cutoff.
        pop_len: Number of individuals in the population.
        decay_start_size: Size at which the decay starts.

    Returns:
        The target probability for that size.
    """
    half_life = decay_start_size * float(alpha) + beta
    hl_1 = gamma * pop_len * math.log(2) / half_life
    hl_2 = math.exp(-math.log(2) * (size - decay_start_size) / half_life)
    return hl_1 * hl_2


def natural_histogram(sizes: list[int], pop_len: int, nb_model: int) -> list[float]:
    """Build the smoothed size distribution of the natural population.

    Each individual contributes to its own size bin and, with smaller
    weights, to the neighboring bins.

    Args:
        sizes: Sizes of the modeled individuals.
        pop_len: Number of individuals in the population.
        nb_model: Number of individuals used to build the model.

    Returns:
        The histogram, scaled to the population size.
    """
    hist: list[float] = [0.0] * (max(sizes) + 3)
    for ind_size in sizes:
        hist[ind_size] += 0.4
        hist[ind_size - 1] += 0.2
        hist[ind_size + 1] += 0.2
        hist[ind_size + 2] += 0.1
        if ind_size - 2 >= 0:
            hist[ind_size - 2] += 0.1
    return [val * pop_len / nb_model for val in hist]


def cutoff_size(natural_pop: list[GPIndividual], pop_len: int, rho: float, min_cutoff: int) -> int:
    """Return the tree size at which the size penalty starts.

    Args:
        natural_pop: Individuals modeling the natural distribution.
        pop_len: Number of individuals in the population.
        rho: Fitness range used to place the cutoff.
        min_cutoff: Absolute minimum cutoff.

    Returns:
        The cutoff size.
    """
    source = [ind for ind in natural_pop if ind.fitness.is_valid()]
    if not source:
        return min_cutoff
    _ = pop_len
    sorted_natural = sorted(source, key=lambda ind: ind.fitness)
    start = max(0, int(len(sorted_natural) * rho - 1))
    cutoff_candidates = sorted_natural[start:]
    if not cutoff_candidates:
        return min_cutoff
    return max(min_cutoff, len(min(cutoff_candidates, key=len)))


def target_histogram(
    natural_hist: list[float], decay_start_size: int, target_prob_fn: Callable[[int], float]
) -> list[float]:
    """Build the desired size distribution of the next generation.

    Bins up to ``decay_start_size`` keep their natural frequency; larger
    bins are replaced by the decaying target.

    Args:
        natural_hist: Natural size distribution.
        decay_start_size: Size at which the decay starts.
        target_prob_fn: Callable returning the target for one size.

    Returns:
        The target histogram, aligned with ``natural_hist``.
    """
    target_hist = []
    for bin_idx in range(len(natural_hist)):
        if bin_idx <= decay_start_size:
            target_hist.append(natural_hist[bin_idx])
        else:
            target_hist.append(target_prob_fn(bin_idx))
    return target_hist


def acceptance(
    natural_hist: list[float], target_hist: list[float], target_prob_fn: Callable[[int], float]
) -> Callable[[int], bool]:
    """Build the size-based acceptance test for one generation.

    Args:
        natural_hist: Natural size distribution.
        target_hist: Desired size distribution.
        target_prob_fn: Callable returning the target for one size.

    Returns:
        A callable that randomly accepts an individual of a given size.
    """
    prob_hist = [t / n if n > 0 else t for n, t in zip(natural_hist, target_hist, strict=False)]

    def accept(size: int) -> bool:
        prob = prob_hist[size] if size < len(prob_hist) else target_prob_fn(size)
        return rng.random() <= prob

    return accept

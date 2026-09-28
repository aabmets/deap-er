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
from collections.abc import Callable, Sequence
from statistics import NormalDist
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = [
    "eliminate_losers",
    "maximize_first_objective",
    "race_z_score",
]


def race_z_score(alpha: float) -> float:
    """Map a two-sided significance level to its normal critical value."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def objective_score(values: Sequence[float], objective: int, maximize: bool) -> float:
    """Return a scalar for ranking on one objective."""
    value = float(values[objective])
    if not math.isfinite(value):
        return float("-inf") if maximize else float("inf")
    return value if maximize else -value


def sample_mean_std(samples: Sequence[Sequence[float]], objective: int) -> tuple[float, float]:
    """Return the sample mean and standard error on one objective."""
    values = [float(sample[objective]) for sample in samples if math.isfinite(sample[objective])]
    if not values:
        return float("nan"), float("inf")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, float("inf")
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance / len(values))


def maximize_first_objective(individuals: Sequence[Individual]) -> bool:
    """Return whether the first fitness objective is maximized."""
    for individual in individuals:
        weights = individual.fitness.weights
        if weights:
            return float(weights[0]) > 0.0
    return False


def challenger_is_not_worse(
    challenger_mean: float,
    challenger_se: float,
    leader_mean: float,
    leader_se: float,
    *,
    z_score: float,
    maximize: bool,
) -> bool:
    """Return whether a challenger is not confidently worse than the leader."""
    if maximize:
        return challenger_mean + z_score * challenger_se >= leader_mean - z_score * leader_se
    return challenger_mean - z_score * challenger_se <= leader_mean + z_score * leader_se


def keep_challenger_after_race(
    challenger_samples: Sequence[Sequence[float]],
    leader_mean: float,
    leader_se: float,
    *,
    z_score: float,
    maximize: bool,
) -> bool:
    """Return whether a challenger survives comparison with the leader."""
    if len(challenger_samples) < 2:
        return True
    challenger_mean, challenger_se = sample_mean_std(challenger_samples, 0)
    if not math.isfinite(challenger_mean):
        return False
    return challenger_is_not_worse(
        challenger_mean,
        challenger_se,
        leader_mean,
        leader_se,
        z_score=z_score,
        maximize=maximize,
    )


def restore_min_survivors(
    kept: list[Individual],
    ranked: Sequence[Individual],
    min_survivors: int,
) -> list[Individual]:
    """Add back ranked challengers until ``min_survivors`` is met."""
    if len(kept) >= min_survivors:
        return kept
    restored = list(kept)
    kept_ids = {id(individual) for individual in kept}
    for challenger in ranked[1:]:
        if id(challenger) not in kept_ids:
            restored.append(challenger)
        if len(restored) >= min_survivors:
            break
    return restored


def eliminate_losers(
    survivors: list[Individual],
    samples: dict[int, list[tuple[float, ...]]],
    *,
    aggregate: Callable[[Sequence[Sequence[float]]], Sequence[float]],
    z_score: float,
    maximize: bool,
    min_survivors: int,
) -> list[Individual]:
    """Drop challengers whose first objective is significantly worse."""
    if len(survivors) <= min_survivors:
        return survivors
    ranked = sorted(
        survivors,
        key=lambda individual: objective_score(aggregate(samples[id(individual)]), 0, maximize),
        reverse=True,
    )
    leader = ranked[0]
    leader_samples = samples[id(leader)]
    if len(leader_samples) < 2:
        return survivors
    leader_mean, leader_se = sample_mean_std(leader_samples, 0)
    if not math.isfinite(leader_mean):
        return survivors
    kept = [leader]
    for challenger in ranked[1:]:
        if keep_challenger_after_race(
            samples[id(challenger)],
            leader_mean,
            leader_se,
            z_score=z_score,
            maximize=maximize,
        ):
            kept.append(challenger)
    return restore_min_survivors(kept, ranked, min_survivors)

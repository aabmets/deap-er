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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from deap_er.private.various.fitness_resample import noisy_draw_key, resample_aggregate

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual
    from deap_er.private.various.eval_cache import EvalCache

__all__: list[str] = ["RaceStopResult", "race_eval_charge", "race_stop"]

_ALPHA_TO_Z: dict[float, float] = {
    0.10: 1.64,
    0.05: 1.96,
    0.01: 2.58,
}


def race_eval_charge(n_individuals: int, n_draws: int = 1) -> int:
    """Return evaluation units for one racing round.

    Args:
        n_individuals: Survivors scored in the round.
        n_draws: Draws per individual (default one).

    Returns:
        ``n_individuals * n_draws``.

    Raises:
        ValueError: If either count is negative.
    """
    if n_individuals < 0 or n_draws < 0:
        raise ValueError("n_individuals and n_draws must be non-negative")
    return n_individuals * n_draws


def race_z_score(alpha: float) -> float:
    """Map a significance level to an approximate normal critical value."""
    if alpha in _ALPHA_TO_Z:
        return _ALPHA_TO_Z[alpha]
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    return 1.96


@dataclass(frozen=True, slots=True)
class RaceStopResult:
    """Outcome of :func:`race_stop`.

    Attributes:
        survivors: Individuals that survived every racing round.
        nevals: Evaluate or cache calls charged across all rounds.
        rounds: Number of resample rounds executed.
    """

    survivors: list[Individual]
    nevals: int
    rounds: int


def score_draw(
    individual: Individual,
    evaluate: Callable[[Individual], Sequence[float]],
    *,
    cache: EvalCache | None,
    key_fn: Callable[[Individual], Any] | None,
    draw: int,
) -> tuple[float, ...]:
    """Score one draw for ``individual`` through cache or ``evaluate``."""
    if cache is not None:
        caller_key = None
        if key_fn is not None:
            caller_key = noisy_draw_key(key_fn(individual), draw)
        cached = cache.evaluate(individual, key=caller_key)
        return tuple(float(value) for value in cached)
    return tuple(float(value) for value in evaluate(individual))


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
        fitness = individual.fitness
        if fitness.is_valid() and fitness.weights:
            return float(fitness.weights[0]) > 0.0
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


def eliminate_losers(
    survivors: list[Individual],
    samples: dict[int, list[tuple[float, ...]]],
    *,
    aggregate: Callable[[Sequence[Sequence[float]]], Sequence[float]],
    alpha: float,
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
    z_score = race_z_score(alpha)
    kept = [leader]
    for challenger in ranked[1:]:
        challenger_samples = samples[id(challenger)]
        if len(challenger_samples) < 2:
            kept.append(challenger)
            continue
        challenger_mean, challenger_se = sample_mean_std(challenger_samples, 0)
        if not math.isfinite(challenger_mean):
            continue
        if challenger_is_not_worse(
            challenger_mean,
            challenger_se,
            leader_mean,
            leader_se,
            z_score=z_score,
            maximize=maximize,
        ):
            kept.append(challenger)
    if len(kept) < min_survivors:
        for challenger in ranked[1:]:
            if challenger not in kept:
                kept.append(challenger)
            if len(kept) >= min_survivors:
                break
    return kept


def race_stop(
    individuals: Sequence[Individual],
    evaluate: Callable[[Individual], Sequence[float]],
    n_rounds: int,
    *,
    cache: EvalCache | None = None,
    key_fn: Callable[[Individual], Any] | None = None,
    alpha: float = 0.05,
    min_survivors: int = 1,
    aggregate: Callable[[Sequence[Sequence[float]]], Sequence[float]] = resample_aggregate,
    write: bool = False,
) -> RaceStopResult:
    """Run an F-Race-shaped stop on noisy fitness draws.

    Each round adds one resample per survivor. After the second round,
    challengers whose first objective is significantly worse than the
    current leader are dropped. Samples are tracked by ``id(individual)``;
    clones share one history unless they are distinct objects.

    Multi-objective racing uses objective zero only. ``evaluate`` stays
    on the caller; this helper does not build a domain metric.

    Args:
        individuals: Candidates to race.
        evaluate: One-draw fitness callable.
        n_rounds: Maximum resample rounds.
        cache: Optional :class:`~deap_er.tools.EvalCache` wrapper.
        key_fn: Optional per-individual cache key fragment.
        alpha: Significance level mapped to a normal critical value.
        min_survivors: Stop eliminating below this count.
        aggregate: Reduces each individual's draw list to one tuple.
        write: When true, write the final aggregate to ``fitness.values``.

    Returns:
        Survivors, total evaluate calls, and rounds executed.

    Raises:
        ValueError: If ``n_rounds`` or ``min_survivors`` is invalid.
    """
    if n_rounds < 0:
        raise ValueError("n_rounds must be non-negative")
    if min_survivors < 1:
        raise ValueError("min_survivors must be at least 1")
    survivors = list(individuals)
    if not survivors or n_rounds == 0:
        return RaceStopResult([], 0, 0)

    samples: dict[int, list[tuple[float, ...]]] = {id(individual): [] for individual in survivors}
    charged = 0
    rounds_run = 0
    maximize = maximize_first_objective(survivors)

    for round_idx in range(n_rounds):
        for individual in survivors:
            draw = score_draw(
                individual,
                evaluate,
                cache=cache,
                key_fn=key_fn,
                draw=round_idx,
            )
            samples[id(individual)].append(draw)
        charged += race_eval_charge(len(survivors), 1)
        rounds_run += 1
        if round_idx >= 1:
            survivors = eliminate_losers(
                survivors,
                samples,
                aggregate=aggregate,
                alpha=alpha,
                maximize=maximize,
                min_survivors=min_survivors,
            )
        if len(survivors) <= min_survivors:
            break

    if write:
        for individual in survivors:
            individual.fitness.values = tuple(aggregate(samples[id(individual)]))

    return RaceStopResult(survivors, charged, rounds_run)

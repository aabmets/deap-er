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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .fitness_resample import cached_draw, resample_aggregate
from .race_rounds import eliminate_losers, maximize_first_objective, race_z_score

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

    from .eval_cache import EvalCache

__all__: list[str] = ["RaceStopResult", "race_eval_charge", "race_stop"]


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
        base = key_fn(individual) if key_fn is not None else None
        return cached_draw(cache, individual, base, draw)
    return tuple(float(value) for value in evaluate(individual))


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
        key_fn: Optional per-individual cache key fragment. Defaults
            to the cache's expression key.
        alpha: Two-sided significance level in ``(0, 1)``, mapped to
            a normal critical value.
        min_survivors: Stop eliminating below this count.
        aggregate: Reduces each individual's draw list to one tuple.
        write: When true, write the final aggregate to ``fitness.values``.

    Returns:
        Survivors, total evaluate calls, and rounds executed.

    Raises:
        ValueError: If ``n_rounds``, ``min_survivors``, or ``alpha`` is
            invalid.
    """
    if n_rounds < 0:
        raise ValueError("n_rounds must be non-negative")
    if min_survivors < 1:
        raise ValueError("min_survivors must be at least 1")
    z_score = race_z_score(alpha)
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
                z_score=z_score,
                maximize=maximize,
                min_survivors=min_survivors,
            )
        if len(survivors) <= min_survivors:
            break

    if write:
        for individual in survivors:
            individual.fitness.values = tuple(aggregate(samples[id(individual)]))

    return RaceStopResult(survivors, charged, rounds_run)

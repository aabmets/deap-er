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

from collections.abc import Callable
from typing import Any, Literal

import numpy

from deap_er.private.strategies.cma_multi_objective import StrategyMultiObjective
from deap_er.private.strategies.cma_one_plus_lambda import StrategyOnePlusLambda
from deap_er.private.strategies.cma_separable import StrategySeparable
from deap_er.private.strategies.cma_standard import Strategy
from deap_er.private.strategies.restart_common import (
    sample_centroid,
    sample_small_lambda,
    sample_small_sigma,
)
from deap_er.private.typedefs import Individual

__all__: list[str] = [
    "apply_strategy_restart",
    "next_bipop_params",
    "next_ipop_params",
    "resize_offsprings",
    "set_strategy_sigma",
    "target_met",
    "validate_lambda_factor",
]


def set_strategy_sigma(strategy: Any, sigma: float) -> None:
    """Set the step size on a CMA strategy."""
    if isinstance(strategy, StrategyMultiObjective):
        strategy.sigmas = [sigma] * len(strategy.parents)
        return
    strategy.sigma = sigma


def resize_offsprings(strategy: Any, lamb: int) -> None:
    """Change the offspring count for a leftover-budget batch.

    ``Strategy`` / ``StrategySeparable`` recombine the best ``mu`` of
    ``lamb`` offspring, so ``mu`` is capped at ``lamb``. MO-CMA selects
    ``mu`` parents from parents plus offspring and keeps its ``mu``.
    """
    kwargs: dict[str, int] = {"offsprings": lamb}
    if isinstance(strategy, StrategyMultiObjective):
        kwargs["survivors"] = int(strategy.mu)
    elif hasattr(strategy, "mu"):
        kwargs["survivors"] = min(int(strategy.mu), lamb)
    strategy.compute_params(**kwargs)


def target_met(
    target_f: float | None,
    weights: tuple[float, ...] | None,
    best_w: float,
) -> bool:
    """Return whether a single-objective weighted best reached ``target_f``."""
    if target_f is None or weights is None or len(weights) != 1:
        return False
    return best_w >= target_f * weights[0]


def apply_strategy_restart(
    strategy: Strategy | StrategySeparable | StrategyOnePlusLambda | StrategyMultiObjective,
    ind_init: Callable[..., Individual],
    *,
    dim: int,
    lamb: int,
    sigma: float,
    restart_centroid: str | Callable[[int], numpy.ndarray],
    initial_center: numpy.ndarray,
    best: Individual | None,
    max_survivors: int | None = None,
) -> None:
    """Reset a wrapped CMA strategy for the next restart.

    For multi-objective strategies, ``survivors`` is
    ``min(max_survivors, lamb)``. Pass the first run's ``mu`` as
    ``max_survivors`` so a small-λ restart does not shrink every later
    run. It defaults to the strategy's current ``mu``.
    """
    center = sample_centroid(
        dim,
        getattr(strategy, "low", None),
        getattr(strategy, "up", None),
        restart_centroid,
        initial_center,
        best,
    )
    if isinstance(strategy, Strategy | StrategySeparable):
        strategy.reset_state(center, sigma, offsprings=lamb)
        return
    if isinstance(strategy, StrategyOnePlusLambda):
        parent = ind_init(center)
        del parent.fitness.values
        strategy.reset_state(parent, sigma, offsprings=lamb)
        return
    cap = strategy.mu if max_survivors is None else max_survivors
    survivors = min(cap, lamb)
    parents = []
    for _ in range(survivors):
        ind = ind_init(
            sample_centroid(
                dim,
                strategy.low,
                strategy.up,
                restart_centroid,
                initial_center,
                best,
            )
        )
        del ind.fitness.values
        parents.append(ind)
    strategy.reset_state(parents, sigma, offsprings=lamb, survivors=survivors)


def validate_lambda_factor(
    lambda_default: int, lambda_factor: float, max_large_restarts: int
) -> None:
    """Reject a ``lambda_factor`` whose large-regime λ can drop below 1.

    Large restarts use ``int(lambda_default * lambda_factor**i)`` for
    ``i`` up to ``max_large_restarts``.

    Args:
        lambda_default: First-run offspring count.
        lambda_factor: Population growth factor per large restart.
        max_large_restarts: Largest exponent ``i``.

    Raises:
        ValueError: If any reachable large-regime λ is less than 1.
    """
    exponents = range(max_large_restarts + 1)
    smallest = min((int(lambda_default * lambda_factor**i) for i in exponents), default=1)
    if smallest < 1:
        raise ValueError(
            f"lambda_factor={lambda_factor} with offsprings={lambda_default} and "
            f"max_large_restarts={max_large_restarts} gives a restart λ of {smallest}; "
            "every restart needs at least 1 offspring."
        )


def next_ipop_params(
    lambda_default: int,
    lambda_factor: float,
    irestart_large: int,
    max_large_restarts: int,
    sigma_large: float,
) -> tuple[int, float, int]:
    """Return the next IPOP offspring count, sigma, and large-restart index."""
    irestart_large = min(irestart_large + 1, max_large_restarts)
    lamb = int(lambda_default * lambda_factor**irestart_large)
    return lamb, sigma_large, irestart_large


def next_bipop_params(
    *,
    lambda_default: int,
    lambda_factor: float,
    lambda_large: int,
    irestart_large: int,
    max_large_restarts: int,
    sigma_large: float,
    restart_count: int,
    evals_used: int,
    budget: int,
    budget_large: int,
    budget_small: int,
) -> tuple[int, float, Literal["large", "small"], int, int]:
    """Return the next BIPOP offspring count, sigma, regime, and large-λ state."""
    force_large = restart_count == 1 or evals_used >= budget * 0.95
    if force_large or budget_small >= budget_large:
        irestart_large = min(irestart_large + 1, max_large_restarts)
        lambda_large = int(lambda_default * lambda_factor**irestart_large)
        return lambda_large, sigma_large, "large", irestart_large, lambda_large
    lamb = sample_small_lambda(lambda_default, lambda_large)
    return lamb, sample_small_sigma(sigma_large), "small", irestart_large, lambda_large

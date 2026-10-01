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
from typing import Any

import numpy

from deap_er.private.strategies.cma_multi_objective import StrategyMultiObjective
from deap_er.private.strategies.cma_one_plus_lambda import StrategyOnePlusLambda
from deap_er.private.strategies.cma_separable import StrategySeparable
from deap_er.private.strategies.cma_standard import Strategy
from deap_er.private.strategies.restart_common import sample_centroid
from deap_er.private.typedefs import Individual

__all__: list[str] = [
    "apply_strategy_restart",
    "resize_offsprings",
    "set_strategy_sigma",
    "target_met",
    "update_or_flag_collapse",
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
    The cap applies to this batch only: the user's pinned
    hyperparameters are kept, with ``offsprings`` set to ``lamb``.
    """
    pins = dict(getattr(strategy, "hyperparams", {}))
    kwargs: dict[str, int] = {"offsprings": lamb}
    if isinstance(strategy, StrategyMultiObjective):
        kwargs["survivors"] = int(strategy.mu)
    elif hasattr(strategy, "mu"):
        kwargs["survivors"] = min(int(strategy.mu), lamb)
    strategy.compute_params(**kwargs)
    strategy.hyperparams = {**pins, "offsprings": lamb}


def update_or_flag_collapse(strategy: Any, population: list[Individual]) -> bool:
    """Update ``strategy`` from ``population``; return whether it collapsed.

    ``Strategy`` and ``StrategySeparable`` raise ``FloatingPointError``
    when the step size collapses, leaving their state unchanged. The
    wrapper ends the run so the next restart replaces that state.
    """
    try:
        strategy.update(population)
    except FloatingPointError:
        return True
    return False


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

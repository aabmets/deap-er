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
from typing import Literal

import numpy

from deap_er.private.strategies.cma_multi_objective import StrategyMultiObjective
from deap_er.private.strategies.cma_one_plus_lambda import StrategyOnePlusLambda
from deap_er.private.strategies.cma_separable import StrategySeparable
from deap_er.private.strategies.cma_standard import Strategy
from deap_er.private.strategies.restart_common import (
    default_lambda,
    max_iter_limit,
    require_stagnation_key,
    scalar_fitness,
    strategy_center,
    strategy_diagnostics,
    strategy_dim,
)
from deap_er.private.strategies.restart_ops import (
    apply_strategy_restart,
    resize_offsprings,
    set_strategy_sigma,
    target_met,
    update_or_flag_collapse,
)
from deap_er.private.strategies.restart_schedule import RestartSchedule
from deap_er.private.strategies.restart_tracker import RunTracker
from deap_er.private.typedefs import Individual

__all__ = ["RestartStrategy"]

StrategyLike = Strategy | StrategySeparable | StrategyOnePlusLambda | StrategyMultiObjective


class RestartStrategy:
    """Wrap a standard, separable, (1+λ), or MO CMA strategy with IPOP or BIPOP restarts.

    See constructor keyword arguments for configuration. ``target_f`` is
    expressed in raw objective space for single-objective runs. The first
    run uses ``sigma_large`` as its initial step size. A run whose step
    size collapses (the inner ``update`` raises ``FloatingPointError``) ends
    and restarts.

    ``stagnation_key`` must return a higher-is-better scalar. It is required
    for multi-objective fitness because there is no default scalarization.
    """

    def __init__(
        self,
        strategy: StrategyLike,
        *,
        mode: Literal["ipop", "bipop"] = "bipop",
        budget: int,
        target_f: float | None = None,
        sigma_large: float = 2.0,
        lambda_factor: float = 2.0,
        max_large_restarts: int = 9,
        max_restarts: int | None = None,
        stagnation_window: int = 20,
        tol_fun: float = 1e-12,
        condition_limit: float = 1e14,
        restart_centroid: str | Callable[[int], numpy.ndarray] = "random",
        stagnation_key: Callable[[Individual], float] | None = None,
    ) -> None:
        """See the class docstring."""
        self.strategy = strategy
        self.budget = budget
        self.target_f = target_f
        self.max_restarts = max_restarts
        self.restart_centroid = restart_centroid
        self.stagnation_key = stagnation_key

        self.dim = strategy_dim(strategy)
        lambda_default = int(getattr(strategy, "lamb", default_lambda(self.dim)))
        self._schedule = RestartSchedule(
            mode, lambda_default, lambda_factor, max_large_restarts, sigma_large
        )
        self._mu_default = int(getattr(strategy, "mu", 1))
        self._restart_count = 0
        self._run_evals = 0
        self._small_run_cap: int | None = None
        self._evals_used = 0
        self._done = False
        self._ind_init: Callable[..., Individual] | None = None
        self._initial_center = strategy_center(strategy)
        self._best: Individual | None = None
        self._best_w: float = float("-inf")
        self._fitness_weights: tuple[float, ...] | None = None
        self._tracker = RunTracker(
            self.dim,
            lambda_default,
            sigma_large,
            stagnation_window=stagnation_window,
            tol_fun=tol_fun,
            condition_limit=condition_limit,
        )
        set_strategy_sigma(self.strategy, sigma_large)
        self._begin_run(lambda_default, sigma_large)

    @property
    def mode(self) -> Literal["ipop", "bipop"]:
        """Restart scheme, ``ipop`` or ``bipop``."""
        return self._schedule.mode

    @property
    def sigma_large(self) -> float:
        """First-run and large-regime step size."""
        return self._schedule.sigma_large

    @property
    def lambda_factor(self) -> float:
        """Population growth factor per large restart."""
        return self._schedule.lambda_factor

    @property
    def max_large_restarts(self) -> int:
        """Cap on the large-restart exponent."""
        return self._schedule.max_large_restarts

    @property
    def evals_used(self) -> int:
        """Total function evaluations consumed so far."""
        return self._evals_used

    @property
    def restart_count(self) -> int:
        """Number of restarts completed."""
        return self._restart_count

    @property
    def regime(self) -> Literal["large", "small"] | None:
        """Active restart regime, or None before the first restart."""
        return self._schedule.regime

    @property
    def best_fitness(self) -> float:
        """Best raw objective seen across all runs for single-objective runs.

        Independent of ``stagnation_key``, which only drives stagnation.
        """
        weights = self._fitness_weights
        if self._evals_used == 0 or weights is None or len(weights) != 1:
            return float("nan")
        weight = float(weights[0])
        return self._best_w / weight if weight != 0 else self._best_w

    def remaining_budget(self) -> int:
        """Function evaluations left before the hard budget is reached."""
        return max(0, self.budget - self._evals_used)

    def generate(self, ind_init: Callable[..., Individual]) -> list[Individual]:
        """Sample offspring from the inner strategy within the eval budget."""
        self._ind_init = ind_init
        remaining = self.remaining_budget()
        if remaining <= 0:
            return []
        requested = self.strategy.lamb
        batch = min(requested, remaining)
        if batch != requested:
            resize_offsprings(self.strategy, batch)
        return self.strategy.generate(ind_init)

    def update(self, population: list[Individual]) -> None:
        """Update the inner strategy and check per-run termination."""
        if not population:
            self._done = True
            return
        if self._fitness_weights is None:
            self._fitness_weights = population[0].fitness.weights
        require_stagnation_key(self._fitness_weights, self.stagnation_key)
        self._tracker.terminate |= update_or_flag_collapse(self.strategy, population)
        self._run_evals += len(population)
        self._evals_used += len(population)
        if len(self._fitness_weights) == 1:
            self._best_w = max(self._best_w, *(float(ind.fitness.wvalues[0]) for ind in population))
        for ind in population:
            if self._best is None or scalar_fitness(ind, self.stagnation_key) > scalar_fitness(
                self._best, self.stagnation_key
            ):
                self._best = ind
        cond, sigma, largest = strategy_diagnostics(self.strategy)
        self._tracker.observe(
            population, self.stagnation_key, condition=cond, sigma=sigma, largest_eig=largest
        )
        if target_met(self.target_f, self._fitness_weights, self._best_w):
            self._done = True
        if self._small_run_cap is not None and self._run_evals >= self._small_run_cap:
            self._tracker.terminate = True
        if self._evals_used >= self.budget:
            self._done = True

    def should_restart(self) -> bool:
        """Return whether the current run ended and a restart is due."""
        if self._done:
            return False
        if not self._tracker.terminate:
            return False
        if self._evals_used >= self.budget:
            return False
        if self.max_restarts is not None and self._restart_count >= self.max_restarts:
            self._done = True
            return False
        return True

    def restart(self) -> None:
        """Finish the current run and launch the next restart."""
        self._schedule.account_run(self._run_evals)
        self._restart_count += 1
        lamb, sigma, self._small_run_cap = self._schedule.next_run(
            self._restart_count, self._evals_used, self.budget
        )
        self._apply_restart(lamb, sigma)
        self._run_evals = 0
        self._begin_run(lamb, sigma)

    def is_done(self) -> bool:
        """Return whether the global budget or target has been reached."""
        return self._done or self._evals_used >= self.budget

    def _begin_run(self, lamb: int, sigma: float) -> None:
        cap = max_iter_limit(self.dim, lamb)
        self._tracker.begin_run(lamb, sigma, max_iter=cap)

    def _apply_restart(self, lamb: int, sigma: float) -> None:
        if self._ind_init is None:
            raise RuntimeError("Call generate before restart.")
        apply_strategy_restart(
            self.strategy,
            self._ind_init,
            dim=self.dim,
            lamb=lamb,
            sigma=sigma,
            restart_centroid=self.restart_centroid,
            initial_center=self._initial_center,
            best=self._best,
            max_survivors=self._mu_default,
        )

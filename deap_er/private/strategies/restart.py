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
from deap_er.private.strategies.cma_standard import Strategy
from deap_er.private.strategies.restart_common import (
    RunTracker,
    default_lambda,
    max_iter_limit,
    scalar_fitness,
    strategy_center,
    strategy_diagnostics,
    strategy_dim,
)
from deap_er.private.strategies.restart_ops import (
    apply_strategy_restart,
    next_bipop_params,
    next_ipop_params,
    resize_offsprings,
    set_strategy_sigma,
    target_met,
)
from deap_er.private.typedefs import Individual

__all__ = ["RestartStrategy"]

StrategyLike = Strategy | StrategyOnePlusLambda | StrategyMultiObjective


class RestartStrategy:
    """Wrap a CMA strategy with IPOP or BIPOP restart scheduling.

    See constructor keyword arguments for configuration. ``target_f`` is
    expressed in raw objective space for single-objective runs. The first
    run uses ``sigma_large`` as its initial step size.
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
        tol_up_sigma: float = 1e20,
        restart_centroid: str | Callable[[int], numpy.ndarray] = "random",
        stagnation_key: Callable[[Individual], float] | None = None,
    ) -> None:
        """See the class docstring."""
        self.strategy = strategy
        self.mode = mode
        self.budget = budget
        self.target_f = target_f
        self.sigma_large = sigma_large
        self.lambda_factor = lambda_factor
        self.max_large_restarts = max_large_restarts
        self.max_restarts = max_restarts
        self.restart_centroid = restart_centroid
        self.stagnation_key = stagnation_key

        self.dim = strategy_dim(strategy)
        self._lambda_default = int(getattr(strategy, "lamb", default_lambda(self.dim)))
        self._lambda_large = self._lambda_default
        self._irestart_large = 0
        self._budget_large = 0
        self._budget_small = 0
        self._run_count = 0
        self._restart_count = 0
        self._regime: Literal["large", "small"] | None = None
        self._run_evals = 0
        self._last_large_run_evals = 0
        self._small_run_cap: int | None = None
        self._evals_used = 0
        self._done = False
        self._ind_init: Callable[..., Individual] | None = None
        self._initial_center = strategy_center(strategy)
        self._best: Individual | None = None
        self._fitness_weights: tuple[float, ...] | None = None
        self._saved_lamb: int | None = None
        self._tracker = RunTracker(
            self.dim,
            self._lambda_default,
            sigma_large,
            stagnation_window=stagnation_window,
            tol_fun=tol_fun,
            condition_limit=condition_limit,
            tol_up_sigma=tol_up_sigma,
        )
        set_strategy_sigma(self.strategy, sigma_large)
        self._begin_run(self._lambda_default, sigma_large)

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
        return self._regime

    @property
    def best_fitness(self) -> float:
        """Best raw objective seen across all runs for single-objective runs."""
        if self._fitness_weights and len(self._fitness_weights) == 1:
            weight = self._fitness_weights[0]
            if weight != 0:
                return float(self._tracker.best_ever / weight)
        return float(self._tracker.best_ever)

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
            self._saved_lamb = requested
            resize_offsprings(self.strategy, batch)
        return self.strategy.generate(ind_init)

    def update(self, population: list[Individual]) -> None:
        """Update the inner strategy and check per-run termination."""
        if not population:
            self._done = True
            return
        if self._fitness_weights is None:
            self._fitness_weights = population[0].fitness.weights
        self.strategy.update(population)
        self._run_evals += len(population)
        self._evals_used += len(population)
        for ind in population:
            if self._best is None or scalar_fitness(ind, self.stagnation_key) > scalar_fitness(
                self._best, self.stagnation_key
            ):
                self._best = ind
        cond, sigma, largest = strategy_diagnostics(self.strategy)
        self._tracker.observe(
            population,
            self.stagnation_key,
            condition=cond,
            sigma=sigma,
            largest_eig=largest,
        )
        if target_met(self.target_f, self._fitness_weights, self._tracker.best_ever):
            self._done = True
        if self._small_run_cap is not None and self._run_evals >= self._small_run_cap:
            self._tracker.terminate = True
        if self._evals_used >= self.budget:
            self._done = True
        if self._saved_lamb is not None and self.remaining_budget() > 0:
            resize_offsprings(self.strategy, self._saved_lamb)
            self._saved_lamb = None

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
        self._account_run_budget()
        self._run_count += 1
        self._restart_count += 1
        if self.mode == "ipop":
            lamb, sigma, self._irestart_large = next_ipop_params(
                self._lambda_default,
                self.lambda_factor,
                self._irestart_large,
                self.max_large_restarts,
                self.sigma_large,
            )
            self._regime = "large"
        else:
            lamb, sigma, self._regime, self._irestart_large, self._lambda_large = (
                next_bipop_params(
                    lambda_default=self._lambda_default,
                    lambda_factor=self.lambda_factor,
                    lambda_large=self._lambda_large,
                    irestart_large=self._irestart_large,
                    max_large_restarts=self.max_large_restarts,
                    sigma_large=self.sigma_large,
                    restart_count=self._restart_count,
                    evals_used=self._evals_used,
                    budget=self.budget,
                    budget_large=self._budget_large,
                    budget_small=self._budget_small,
                )
            )
        self._small_run_cap = (
            max(1, self._last_large_run_evals // 2) if self._regime == "small" else None
        )
        self._apply_restart(lamb, sigma)
        self._run_evals = 0
        self._saved_lamb = None
        self._begin_run(lamb, sigma)

    def is_done(self) -> bool:
        """Return whether the global budget or target has been reached."""
        return self._done or self._evals_used >= self.budget

    def _begin_run(self, lamb: int, sigma: float) -> None:
        cap = max_iter_limit(self.dim, lamb)
        self._tracker.begin_run(lamb, sigma, max_iter=cap)

    def _account_run_budget(self) -> None:
        if self._run_count == 0:
            return
        if self._regime == "small":
            self._budget_small += self._run_evals
        else:
            self._budget_large += self._run_evals
            self._last_large_run_evals = self._run_evals

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
        )

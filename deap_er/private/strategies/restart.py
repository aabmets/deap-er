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
    sample_centroid,
    sample_small_lambda,
    sample_small_sigma,
    scalar_fitness,
    strategy_center,
    strategy_diagnostics,
    strategy_dim,
)
from deap_er.private.typedefs import Individual

__all__ = ["RestartStrategy"]

StrategyLike = Strategy | StrategyOnePlusLambda | StrategyMultiObjective


class RestartStrategy:
    """Wrap a CMA strategy with IPOP or BIPOP restart logic.

    Args:
        strategy: ``Strategy``, ``StrategyOnePlusLambda``, or
            ``StrategyMultiObjective`` instance to wrap.
        mode: ``bipop`` (default) or ``ipop``.
        budget: Total function-evaluation budget across all runs.
        target_f: Optional target objective; stop when best is at or
            below this value.
        sigma_large: Initial step size for large / IPOP restarts.
        lambda_factor: Multiplier applied to offspring count on each
            large restart.
        max_large_restarts: Cap on large-regime doublings.
        max_restarts: Optional cap on the number of restarts.
        stagnation_window: Median comparison window for stagnation.
        tol_fun: Relative best-fitness improvement threshold.
        condition_limit: Restart when ``Strategy.cond`` exceeds this.
        tol_up_sigma: Restart when step size grows too fast vs
            covariance.
        restart_centroid: ``random``, ``initial``, ``best``, or a
            callable ``(dim) -> vector``.
        stagnation_key: Optional scalarizer for multi-objective runs.
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
        self._tracker = RunTracker(
            self.dim,
            self._lambda_default,
            sigma_large,
            stagnation_window=stagnation_window,
            tol_fun=tol_fun,
            condition_limit=condition_limit,
            tol_up_sigma=tol_up_sigma,
        )
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
        """Best scalar objective seen across all runs."""
        return self._tracker.best_ever

    def generate(self, ind_init: Callable[..., Individual]) -> list[Individual]:
        """Sample offspring from the inner strategy."""
        self._ind_init = ind_init
        return self.strategy.generate(ind_init)

    def update(self, population: list[Individual]) -> None:
        """Update the inner strategy and check per-run termination."""
        self.strategy.update(population)
        self._run_evals += len(population)
        self._evals_used += len(population)
        for ind in population:
            if self._best is None or scalar_fitness(ind, self.stagnation_key) < scalar_fitness(
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
        if self.target_f is not None and self._tracker.best_ever <= self.target_f:
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
        self._account_run_budget()
        self._run_count += 1
        self._restart_count += 1
        if self.mode == "ipop":
            lamb, sigma = self._next_ipop_params()
            self._regime = "large"
        else:
            lamb, sigma, self._regime = self._next_bipop_params()
        self._small_run_cap = (
            max(1, self._last_large_run_evals // 2) if self._regime == "small" else None
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

    def _account_run_budget(self) -> None:
        if self._run_count == 0:
            return
        if self._regime == "small":
            self._budget_small += self._run_evals
        else:
            self._budget_large += self._run_evals
            self._last_large_run_evals = self._run_evals

    def _next_ipop_params(self) -> tuple[int, float]:
        self._irestart_large = min(self._irestart_large + 1, self.max_large_restarts)
        lamb = int(self._lambda_default * self.lambda_factor**self._irestart_large)
        return lamb, self.sigma_large

    def _next_bipop_params(self) -> tuple[int, float, Literal["large", "small"]]:
        force_large = self._restart_count == 1 or self._evals_used >= self.budget * 0.95
        if force_large or self._budget_small >= self._budget_large:
            self._irestart_large = min(self._irestart_large + 1, self.max_large_restarts)
            self._lambda_large = int(
                self._lambda_default * self.lambda_factor**self._irestart_large
            )
            return self._lambda_large, self.sigma_large, "large"
        lamb = sample_small_lambda(self._lambda_default, self._lambda_large)
        return lamb, sample_small_sigma(), "small"

    def _apply_restart(self, lamb: int, sigma: float) -> None:
        if self._ind_init is None:
            raise RuntimeError("Call generate before restart.")
        center = sample_centroid(
            self.dim,
            getattr(self.strategy, "low", None),
            getattr(self.strategy, "up", None),
            self.restart_centroid,
            self._initial_center,
            self._best,
        )
        if isinstance(self.strategy, Strategy):
            self.strategy.reset_state(center, sigma, offsprings=lamb)
        elif isinstance(self.strategy, StrategyOnePlusLambda):
            parent = self._ind_init(center)
            del parent.fitness.values
            self.strategy.reset_state(parent, sigma, offsprings=lamb)
        else:
            parents = []
            for _ in range(self.strategy.mu):
                ind = self._ind_init(
                    sample_centroid(
                        self.dim,
                        self.strategy.low,
                        self.strategy.up,
                        self.restart_centroid,
                        self._initial_center,
                        self._best,
                    )
                )
                del ind.fitness.values
                parents.append(ind)
            survivors = min(self.strategy.mu, lamb)
            self.strategy.reset_state(parents, sigma, offsprings=lamb, survivors=survivors)

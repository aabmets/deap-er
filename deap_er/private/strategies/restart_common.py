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
from math import ceil, log, sqrt
from typing import TYPE_CHECKING, Any

import numpy

from deap_er.private.strategies.common import _bound_arrays
from deap_er.private.various.rng import rng

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = [
    "RunTracker",
    "default_lambda",
    "max_iter_limit",
    "sample_centroid",
    "sample_small_lambda",
    "sample_small_sigma",
    "scalar_fitness",
    "stagnation_window_size",
    "strategy_center",
    "strategy_sigma",
    "strategy_diagnostics",
    "strategy_dim",
]


def default_lambda(dim: int) -> int:
    """Return the default CMA offspring count for dimension ``dim``."""
    return int(4 + 3 * log(dim))


def max_iter_limit(dim: int, lamb: int, base: int = 1000, scale: float = 500.0) -> int:
    """Return the per-run generation cap used by BBOB-style restarts."""
    return int(base + scale * (dim + 3) ** 2 / sqrt(lamb))


def stagnation_window_size(gen: int, dim: int, lamb: int) -> int:
    """Return the stagnation history length for generation ``gen``."""
    return int(ceil(0.2 * gen + 120 + 30 * dim / lamb))


def sample_centroid(
    dim: int,
    low: Any,
    up: Any,
    mode: str | Callable[[int], numpy.ndarray],
    initial: numpy.ndarray,
    best: Individual | None,
) -> numpy.ndarray:
    """Draw a restart centroid according to ``mode``.

    Args:
        dim: Search-space dimension.
        low: Optional lower box bound.
        up: Optional upper box bound.
        mode: ``random``, ``initial``, ``best``, or a callable.
        initial: Centroid from the first run.
        best: Best individual so far, if any.

    Returns:
        A length-``dim`` centroid vector.
    """
    if callable(mode):
        return numpy.asarray(mode(dim), dtype=float)  # ty: ignore[call-top-callable]
    if mode == "initial":
        return numpy.asarray(initial, dtype=float)
    if mode == "best" and best is not None:
        return numpy.asarray(best, dtype=float)
    bounds = _bound_arrays(low, up, dim)
    if bounds is None:
        return numpy.array([rng.uniform(-5.0, 5.0) for _ in range(dim)])
    low_arr, up_arr = bounds
    return numpy.array([rng.uniform(lo, hi) for lo, hi in zip(low_arr, up_arr, strict=True)])


def sample_small_lambda(lambda_default: int, lambda_large: int) -> int:
    """Sample a BIPOP small-regime offspring count."""
    ratio = lambda_large / (2.0 * lambda_default)
    return max(1, int(lambda_default * ratio ** rng.random()))


def sample_small_sigma() -> float:
    """Sample a BIPOP small-regime initial step size."""
    return float(2.0 * 10.0 ** (-2.0 * rng.random()))


def scalar_fitness(ind: Individual, key: Callable[[Individual], float] | None = None) -> float:
    """Return a weighted scalar for stagnation checks (higher is better).

    For single-objective fitness, uses ``wvalues[0]``. For multi-objective
    fitness, ``key`` must be supplied — there is no default scalarization.

    Args:
        ind: Evaluated individual.
        key: Optional callable returning a higher-is-better stagnation scalar.

    Raises:
        ValueError: If ``key`` is omitted for multi-objective fitness.
    """
    if key is not None:
        return float(key(ind))
    wvalues = ind.fitness.wvalues
    if len(wvalues) == 1:
        return float(wvalues[0])
    raise ValueError(
        "multi-objective stagnation requires an explicit stagnation_key "
        "callable returning a higher-is-better scalar"
    )


class RunTracker:
    """Track per-run fitness history and termination criteria."""

    def __init__(
        self,
        dim: int,
        lamb: int,
        sigma0: float,
        stagnation_window: int = 20,
        tol_fun: float = 1e-12,
        condition_limit: float = 1e14,
        tol_up_sigma: float = 1e20,
        max_iter: int | None = None,
    ) -> None:
        """See class attributes; parameters mirror ``RestartStrategy`` termination."""
        self.dim = dim
        self.lamb = lamb
        self.sigma0 = sigma0
        self.stagnation_window = stagnation_window
        self.tol_fun = tol_fun
        self.condition_limit = condition_limit
        self.tol_up_sigma = tol_up_sigma
        self.max_iter = max_iter
        self.gen = 0
        self.best_history: list[float] = []
        self.median_history: list[float] = []
        self.best_ever = -numpy.inf
        self.terminate = False

    def begin_run(self, lamb: int, sigma0: float, max_iter: int | None = None) -> None:
        """Reset counters for a new CMA run."""
        self.lamb = lamb
        self.sigma0 = sigma0
        self.max_iter = max_iter
        self.gen = 0
        self.best_history.clear()
        self.median_history.clear()
        self.terminate = False

    def observe(
        self,
        population: Sequence[Individual],
        fitness_key: Callable[[Individual], float] | None = None,
        condition: float | None = None,
        sigma: float | None = None,
        largest_eig: float | None = None,
    ) -> None:
        """Record one generation and update termination flags."""
        values = [scalar_fitness(ind, fitness_key) for ind in population]
        best = max(values)
        median = float(numpy.median(values))
        self.gen += 1
        self.best_history.append(best)
        self.median_history.append(median)
        self.best_ever = max(self.best_ever, best)
        if self.max_iter is not None and self.gen >= self.max_iter:
            self.terminate = True
            return
        if self._stagnated():
            self.terminate = True
            return
        if self._tol_fun_hit():
            self.terminate = True
            return
        if condition is not None and condition > self.condition_limit:
            self.terminate = True
            return
        if (
            sigma is not None
            and largest_eig is not None
            and sigma / self.sigma0 > self.tol_up_sigma * sqrt(largest_eig)
        ):
            self.terminate = True

    def _stagnated(self) -> bool:
        window = stagnation_window_size(self.gen, self.dim, self.lamb)
        need = window
        if len(self.best_history) < need:
            return False
        span = self.stagnation_window
        if window < 2 * span:
            return False
        best_slice = self.best_history[-window:]
        med_slice = self.median_history[-window:]
        old_best = numpy.median(best_slice[:span])
        new_best = numpy.median(best_slice[-span:])
        old_med = numpy.median(med_slice[:span])
        new_med = numpy.median(med_slice[-span:])
        return bool(new_best <= old_best and new_med <= old_med)

    def _tol_fun_hit(self) -> bool:
        if len(self.best_history) < 2:
            return False
        span = min(self.stagnation_window, len(self.best_history))
        old = self.best_history[-span]
        new = self.best_history[-1]
        denom = max(abs(old), abs(new), 1e-20)
        return abs(old - new) / denom < self.tol_fun


def strategy_dim(strategy: Any) -> int:
    """Return the search-space dimension of a CMA strategy."""
    return int(strategy.dim)


def strategy_center(strategy: Any) -> numpy.ndarray:
    """Return the initial search center of a CMA strategy."""
    if hasattr(strategy, "centroid"):
        return numpy.asarray(strategy.centroid, dtype=float)
    if hasattr(strategy, "parent"):
        return numpy.asarray(strategy.parent, dtype=float)
    return numpy.asarray(strategy.parents[0], dtype=float)


def strategy_sigma(strategy: Any) -> float:
    """Return the current step size of a CMA strategy."""
    if hasattr(strategy, "sigma") and not hasattr(strategy, "sigmas"):
        return float(strategy.sigma)
    if hasattr(strategy, "sigmas"):
        return float(strategy.sigmas[0])
    return float(strategy.sigma)


def strategy_diagnostics(strategy: Any) -> tuple[float | None, float | None, float | None]:
    """Return optional condition number, sigma, and largest covariance eigenvalue."""
    if hasattr(strategy, "cond"):
        largest = float(strategy.diag_d[-1] ** 2) if len(strategy.diag_d) else 1.0
        return float(strategy.cond), float(strategy.sigma), largest
    if hasattr(strategy, "parent"):
        return None, float(strategy.sigma), None
    return None, float(strategy.sigmas[0]), None

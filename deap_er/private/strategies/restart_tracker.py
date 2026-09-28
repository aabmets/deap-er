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
from math import ceil, sqrt
from typing import TYPE_CHECKING

import numpy

from deap_er.private.strategies.restart_common import scalar_fitness, stagnation_window_size

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = ["RunTracker"]


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
        need = 10 + int(ceil(30 * self.dim / max(self.lamb, 1)))
        if len(self.best_history) < need:
            return False
        window = self.best_history[-need:]
        return max(window) - min(window) < self.tol_fun

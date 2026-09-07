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

from collections.abc import Callable, Iterable
from math import log, sqrt
from typing import TYPE_CHECKING, Any

import numpy

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

from .common import sample_offspring, update_bound_attrs

__all__ = ["StrategySeparable"]


class StrategySeparable:
    """Separable CMA-ES with a diagonal covariance (Ros and Hansen, 2008).

    Learns one variance per gene. Memory and the generate/update step
    are ``O(n)``. There is no learned correlation. Default
    ``rank_one`` and ``rank_mu`` are the ``Strategy`` defaults scaled
    by ``(n + 2) / 3``. Step-size uses the same cumulative step-size
    adaptation as ``Strategy``. Box bounds and the rest of the
    keyword surface match ``Strategy``, except ``cm_init`` is a
    length-``n`` variance vector (default ones), not an ``n``-by-``n``
    matrix.

    Args:
        centroid: Starting point of the search distribution.
        sigma: Initial standard deviation of the distribution.
        **kwargs: Optional strategy parameters. Shared names follow
            ``Strategy`` (``offsprings``, ``survivors``, ``weights``,
            ``cm_cum``, ``ss_cum``, ``ss_dmp``, ``rank_one``,
            ``rank_mu``, ``low``, ``up``, ``bound_mode``,
            ``resample_limit``). ``cm_init`` must be a length-``n``
            variance vector or a scalar broadcast to ``n``.

    Raises:
        RuntimeError: If ``weights`` is not ``superlinear``,
            ``linear``, or ``equal``.
        ValueError: If ``cm_init`` is not a length-``n`` vector, or
            if box-bound kwargs are invalid.
    """

    def __init__(self, centroid: Iterable[float], sigma: float, **kwargs: Any) -> None:
        """See the class docstring."""
        self.update_count = 0
        self.centroid = numpy.array(centroid)
        self.sigma = sigma
        self.dim = len(self.centroid)
        self.pc = numpy.zeros(self.dim)
        self.ps = numpy.zeros(self.dim)
        temp = 1 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim**2)
        self.chi_n = sqrt(self.dim) * temp
        self.lamb: int
        self.mu: int
        self.weights: numpy.ndarray
        self.mu_eff: float
        self.rank_one: float
        self.rank_mu: float
        self.ss_cum: float
        self.ss_dmp: float
        self.cm_cum: float
        self.big_c: numpy.ndarray
        self.diag_d: numpy.ndarray
        self.cond: float
        self.low: Any
        self.up: Any
        self.bound_mode: str
        self.resample_limit: int
        self.compute_params(**kwargs)

    def compute_params(self, **kwargs: Any) -> None:
        """Recompute strategy parameters from ``kwargs``.

        Called from the constructor. Call again if ``offsprings``
        changes during evolution.

        Args:
            **kwargs: Optional strategy parameters. See the class
                docstring.

        Raises:
            RuntimeError: If ``weights`` is not ``superlinear``,
                ``linear``, or ``equal``.
            ValueError: If ``cm_init`` is not a length-``n`` vector.
        """
        default = int(4 + 3 * log(self.dim))
        self.lamb = int(kwargs.get("offsprings", default))
        default = int(self.lamb / 2)
        self.mu = int(kwargs.get("survivors", default))
        r_weights = kwargs.get("weights", "superlinear")
        if r_weights == "superlinear":
            temp_1 = numpy.log(numpy.arange(1, self.mu + 1))
            self.weights = log(self.mu + 0.5) - temp_1
        elif r_weights == "linear":
            temp_1 = numpy.arange(1, self.mu + 1)
            self.weights = self.mu + 0.5 - temp_1
        elif r_weights == "equal":
            self.weights = numpy.ones(self.mu)
        else:
            raise RuntimeError(f"Unknown weights : {r_weights}")
        self.weights = numpy.asarray(self.weights, dtype=float)
        self.weights /= sum(self.weights)
        self.mu_eff = 1.0 / sum(self.weights**2)
        scale = (self.dim + 2.0) / 3.0
        default = 2.0 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        if "rank_one" in kwargs:
            self.rank_one = float(kwargs["rank_one"])
        else:
            self.rank_one = float(default * scale)
        temp_1 = self.mu_eff - 2.0 + 1.0 / self.mu_eff
        temp_2 = (self.dim + 2.0) ** 2 + self.mu_eff
        default = 2.0 * temp_1 / temp_2
        if "rank_mu" in kwargs:
            self.rank_mu = float(kwargs["rank_mu"])
        else:
            self.rank_mu = float(default * scale)
        self.rank_mu = min(1 - self.rank_one, self.rank_mu)
        default = (self.mu_eff + 2.0) / (self.dim + self.mu_eff + 3.0)
        self.ss_cum = float(kwargs.get("ss_cum", default))
        temp_1 = sqrt((self.mu_eff - 1.0) / (self.dim + 1.0))
        temp_2 = max(0.0, temp_1 - 1.0)
        default = 1.0 + 2.0 * temp_2 + self.ss_cum
        self.ss_dmp = float(kwargs.get("ss_dmp", default))
        default = 4.0 / (self.dim + 4.0)
        self.cm_cum = float(kwargs.get("cm_cum", default))
        if not hasattr(self, "big_c") or "cm_init" in kwargs:
            self.big_c = _variance_vector(kwargs.get("cm_init", numpy.ones(self.dim)), self.dim)
            self.diag_d = numpy.sqrt(self.big_c)
            self.cond = float(numpy.max(self.big_c) / numpy.min(self.big_c))
        update_bound_attrs(self, kwargs)

    def reset_state(
        self,
        centroid: Iterable[float],
        sigma: float,
        **kwargs: Any,
    ) -> None:
        """Reset mutable CMA state for a restart."""
        self.centroid = numpy.asarray(centroid, dtype=float)
        self.sigma = sigma
        self.pc = numpy.zeros(self.dim)
        self.ps = numpy.zeros(self.dim)
        self.update_count = 0
        self.compute_params(cm_init=numpy.ones(self.dim), **kwargs)

    def generate(self, ind_init: Callable[..., Individual]) -> list[Individual]:
        """Sample ``offsprings`` individuals from the current distribution.

        Args:
            ind_init: Callable that turns a sampled vector into an
                individual.

        Returns:
            Newly sampled individuals.
        """
        return sample_offspring(
            self.centroid,
            self.sigma,
            self.diag_d,
            self.lamb,
            self.dim,
            ind_init,
            low=self.low,
            up=self.up,
            bound_mode=self.bound_mode,
            resample_limit=self.resample_limit,
        )

    def update(self, population: list[Individual]) -> None:
        """Update centroid, step-size, and diagonal covariance.

        Individuals are ranked by fitness. The best ``survivors``
        members drive the update.

        Args:
            population: Evaluated individuals from ``generate``.
        """
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        old_centroid = self.centroid
        self.centroid = numpy.dot(self.weights, numpy.asarray(population[0 : self.mu]))
        c_diff = self.centroid - old_centroid
        temp_1 = sqrt(self.ss_cum * (2 - self.ss_cum) * self.mu_eff)
        self.ps = (1 - self.ss_cum) * self.ps + temp_1 / self.sigma * (c_diff / self.diag_d)
        temp_1 = sqrt(1.0 - (1.0 - self.ss_cum) ** (2.0 * (self.update_count + 1.0)))
        temp_2 = numpy.linalg.norm(self.ps) / temp_1 / self.chi_n < (1.4 + 2.0 / (self.dim + 1.0))
        hsig = float(temp_2)
        temp_1 = sqrt(self.cm_cum * (2 - self.cm_cum) * self.mu_eff)
        self.pc = (1 - self.cm_cum) * self.pc + hsig * temp_1 / self.sigma * c_diff
        ar_tmp = numpy.asarray(population[0 : self.mu]) - old_centroid
        temp_0 = (1 - hsig) * self.rank_one * self.cm_cum * (2 - self.cm_cum)
        temp_1 = 1 - self.rank_one - self.rank_mu + temp_0
        self.big_c = (
            temp_1 * self.big_c
            + self.rank_one * self.pc**2
            + self.rank_mu * numpy.dot(self.weights, ar_tmp**2) / self.sigma**2
        )
        self.big_c = numpy.maximum(self.big_c, numpy.finfo(float).tiny)
        temp = numpy.linalg.norm(self.ps) / self.chi_n - 1.0
        self.sigma *= numpy.exp(temp * self.ss_cum / self.ss_dmp)
        self.diag_d = numpy.sqrt(self.big_c)
        self.cond = float(numpy.max(self.big_c) / numpy.min(self.big_c))
        self.update_count += 1


def _variance_vector(cm_init: Any, dim: int) -> numpy.ndarray:
    """Return a length-``dim`` positive variance vector from ``cm_init``."""
    values = numpy.asarray(cm_init, dtype=float)
    if values.ndim == 0:
        values = numpy.full(dim, float(values))
    if values.ndim != 1 or values.size != dim:
        raise ValueError("cm_init must be a length-n variance vector.")
    return values

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

__all__ = ["Strategy"]


class Strategy:
    """Standard Covariance Matrix Adaptation evolution strategy.

    Hansen's ``chiN`` and ``diag(D)`` are stored as ``chi_n`` and
    ``diag_d``. ``chi_n`` is the expected norm of an
    ``N``-dimensional standard normal vector. ``diag_d`` is the
    diagonal of ``D``, the square-root eigenvalues of the covariance
    ``C``.

    Args:
        centroid: Starting point of the search distribution.
        sigma: Initial standard deviation of the distribution.
        **kwargs: Optional strategy parameters. See the table below.

    .. dropdown:: Table of Kwargs
       :margin: 0 5 0 0

       * offsprings - *(int)*
          * The number of children to produce at each generation.
          * *Default:* ``int(4 + 3 * log(len(centroid)))``
       * survivors - *(int)*
          * The number of children to keep as parents for the next generation.
          * *Default:* ``int(offsprings / 2)``
       * weights - *(str)*
          * Recombination weights. One of ``superlinear``, ``linear``,
            or ``equal``.
          * *Default:* ``'superlinear'``
       * cm_init - *(numpy.ndarray)*
          * The initial covariance matrix of the distribution.
          * *Default:* ``numpy.identity(len(centroid))``
       * cm_cum - *(float)*
          * Cumulation constant of the covariance matrix.
          * *Default:* ``4 / (len(centroid) + 4)``
       * ss_cum - *(float)*
          * Cumulation constant of the step-size.
          * *Default:* ``(mueff + 2) / (len(centroid) + mueff + 3)``
       * ss_dmp - *(float)*
          * Damping of the step-size.
          * *Default:* ``1 + 2 * max(0, sqrt((mueff - 1) / (len(centroid) + 1)) - 1) + ss_cum``
       * rank_one - *(float)*
          * Learning rate for rank-one update.
          * *Default:* ``2 / ((len(centroid) + 1.3) ** 2 + mueff)``
       * rank_mu - *(float)*
          * Learning rate for rank-mu update.
          * *Default:* ``2 * (mueff - 2 + 1 / mueff) / ((len(centroid) + 2) ** 2 + mueff)``
       * low, up - *(float or sequence)*
          * Optional box bounds on generated individuals.
       * bound_mode - *(str)*
          * ``clip`` (default) or ``resample``. Both are
            constraint-handling approximations; the update treats the
            repaired point as the sample.
       * resample_limit - *(int)*
          * Failed redraws before clipping one sample. *Default:* ``100``
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
        self.big_b: numpy.ndarray
        self.big_bd: numpy.ndarray
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
        """
        default = int(4 + 3 * log(self.dim))
        self.lamb = int(kwargs.get("offsprings", default))

        default = int(self.lamb / 2)
        self.mu = int(kwargs.get("survivors", default))

        default = "superlinear"
        r_weights = kwargs.get("weights", default)
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

        default = 2.0 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.rank_one = float(kwargs.get("rank_one", default))

        temp_1 = self.mu_eff - 2.0 + 1.0 / self.mu_eff
        temp_2 = (self.dim + 2.0) ** 2 + self.mu_eff
        default = 2.0 * temp_1 / temp_2
        self.rank_mu = float(kwargs.get("rank_mu", default))
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
            self.big_c = kwargs.get("cm_init", numpy.identity(self.dim))
            self.diag_d, self.big_b = numpy.linalg.eigh(self.big_c)
            indx = numpy.argsort(self.diag_d)
            self.cond = self.diag_d[indx[-1]] / self.diag_d[indx[0]]
            self.diag_d = self.diag_d[indx] ** 0.5
            self.big_b = self.big_b[:, indx]
            self.big_bd = self.big_b * self.diag_d
        update_bound_attrs(self, kwargs)

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
            self.big_bd,
            self.lamb,
            self.dim,
            ind_init,
            low=self.low,
            up=self.up,
            bound_mode=self.bound_mode,
            resample_limit=self.resample_limit,
        )

    def update(self, population: list[Individual]) -> None:
        """Update centroid, step-size, and covariance from ``population``.

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
        temp_2 = numpy.dot(self.big_b, (1.0 / self.diag_d) * numpy.dot(self.big_b.T, c_diff))
        self.ps = (1 - self.ss_cum) * self.ps + temp_1 / self.sigma * temp_2

        temp_1 = sqrt(1.0 - (1.0 - self.ss_cum) ** (2.0 * (self.update_count + 1.0)))
        temp_2 = numpy.linalg.norm(self.ps) / temp_1 / self.chi_n < (1.4 + 2.0 / (self.dim + 1.0))
        hsig = float(temp_2)

        temp_1 = sqrt(self.cm_cum * (2 - self.cm_cum) * self.mu_eff)
        self.pc = (1 - self.cm_cum) * self.pc + hsig * temp_1 / self.sigma * c_diff

        ar_tmp = population[0 : self.mu] - old_centroid
        temp_0 = (1 - hsig) * self.rank_one * self.cm_cum * (2 - self.cm_cum)
        temp_1 = 1 - self.rank_one - self.rank_mu + temp_0
        temp_2 = numpy.outer(self.pc, self.pc)
        temp_3 = numpy.dot((self.weights * ar_tmp.T), ar_tmp)
        self.big_c = (
            temp_1 * self.big_c + self.rank_one * temp_2 + self.rank_mu * temp_3 / self.sigma**2
        )

        temp = numpy.linalg.norm(self.ps) / self.chi_n - 1.0
        self.sigma *= numpy.exp(temp * self.ss_cum / self.ss_dmp)

        self.diag_d, self.big_b = numpy.linalg.eigh(self.big_c)
        indx = numpy.argsort(self.diag_d)

        self.cond = self.diag_d[indx[-1]] / self.diag_d[indx[0]]

        self.diag_d = self.diag_d[indx] ** 0.5
        self.big_b = self.big_b[:, indx]
        self.big_bd = self.big_b * self.diag_d

        self.update_count += 1

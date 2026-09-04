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
from deap_er.base.dtypes import *
from typing import Any
from collections.abc import Callable
from math import sqrt, exp
import numpy
import copy


__all__ = ["StrategyOnePlusLambda"]


class StrategyOnePlusLambda:
    """One-plus-lambda Covariance Matrix Adaptation evolution strategy.

    Args:
        parent: Starting individual. Must have a fitness attribute.
        sigma: Initial standard deviation of the distribution.
        **kwargs: Optional strategy parameters. See the table below.

    Raises:
        TypeError: If ``parent`` has no fitness attribute.

    .. dropdown:: Table of Kwargs
       :margin: 0 5 0 0

       * offsprings - *(int)*
          * The number of children to produce at each generation.
          * *Default:* ``1``
       * ss_dmp - *(float)*
          * Damping of the step-size.
          * *Default:* ``1.0 + len(parent) / (2.0 * offsprings)``
       * th_cum - *(float)*
          * Time horizon of the cumulative contribution.
          * *Default:* ``2.0 / (len(parent) + 2.0)``
       * tgt_sr - *(float)*
          * Target success rate.
          * *Default:* ``1.0 / (5 + sqrt(offsprings) / 2.0)``
       * thresh_sr - *(float)*
          * Threshold success rate.
          * *Default:* ``0.44``
       * ss_learn_rate - *(float)*
          * Learning rate of the step-size.
          * *Default:* ``tgt_sr * offsprings / (2.0 + tgt_sr * offsprings)``
       * cm_learn_rate - *(float)*
          * Learning rate of the covariance matrix.
          * *Default:* ``2.0 / (len(parent) ** 2 + 6.0)``
    """

    def __init__(self, parent: Individual, sigma: float, **kwargs: Any) -> None:
        """See the class docstring."""
        if not hasattr(parent, "fitness"):
            raise TypeError("The parent must have a fitness attribute.")

        self.parent = parent
        self.sigma = sigma

        self.dim = len(self.parent)
        self.big_c = numpy.identity(self.dim)
        self.big_a = numpy.identity(self.dim)
        self.pc = numpy.zeros(self.dim)

        self.lamb = None
        self.thresh_sr = None
        self.ss_dmp = None
        self.tgt_sr = None
        self.ss_learn_rate = None
        self.th_cum = None
        self.cm_learn_rate = None
        self.psucc = None

        self.compute_params(**kwargs)

    def compute_params(self, **kwargs: Any) -> None:
        """Recompute strategy parameters from ``kwargs``.

        Called from the constructor. Call again if ``offsprings``
        changes during evolution.

        Args:
            **kwargs: Optional strategy parameters. See the class
                docstring.
        """
        self.lamb = kwargs.get("offsprings", 1)
        self.thresh_sr = kwargs.get("thresh_sr", 0.44)

        default = 1.0 + self.dim / (2.0 * self.lamb)
        self.ss_dmp = kwargs.get("ss_dmp", default)

        default = 1.0 / (5 + sqrt(self.lamb) / 2.0)
        self.tgt_sr = kwargs.get("tgt_sr", default)

        default = self.tgt_sr * self.lamb / (2 + self.tgt_sr * self.lamb)
        self.ss_learn_rate = kwargs.get("ss_learn_rate", default)

        default = 2.0 / (self.dim + 2.0)
        self.th_cum = kwargs.get("th_cum", default)

        default = 2.0 / (self.dim**2 + 6.0)
        self.cm_learn_rate = kwargs.get("cm_learn_rate", default)

        self.psucc = self.tgt_sr

    def generate(self, ind_init: Callable[..., Individual]) -> list[Individual]:
        """Sample ``offsprings`` individuals around the current parent.

        Args:
            ind_init: Callable that turns a sampled vector into an
                individual.

        Returns:
            Newly sampled individuals.
        """
        arz = numpy.random.standard_normal((self.lamb, self.dim))
        arz = self.parent + self.sigma * numpy.dot(arz, self.big_a.T)
        return list(map(ind_init, arz))

    def update(self, population: list[Individual]) -> None:
        """Update parent, step-size, and covariance from ``population``.

        The parent is replaced when a better offspring exists. Success
        rate drives the step-size; a successful replacement also
        updates the covariance.

        Args:
            population: Evaluated individuals from ``generate``.
        """
        if hasattr(self.parent, "fitness"):
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            lambda_succ = sum(self.parent.fitness <= ind.fitness for ind in population)
            psucc = float(lambda_succ) / self.lamb
            self.psucc = (1 - self.ss_learn_rate) * self.psucc + self.ss_learn_rate * psucc

            if self.parent.fitness <= population[0].fitness:
                x_step = (population[0] - numpy.array(self.parent)) / self.sigma
                self.parent = copy.deepcopy(population[0])
                if self.psucc < self.thresh_sr:
                    temp_1 = sqrt(self.th_cum * (2 - self.th_cum))
                    self.pc = (1 - self.th_cum) * self.pc + temp_1 * x_step
                    temp_1 = numpy.outer(self.pc, self.pc)
                    self.big_c = (1 - self.cm_learn_rate) * self.big_c + self.cm_learn_rate * temp_1
                else:
                    self.pc = (1 - self.th_cum) * self.pc
                    temp_1 = numpy.outer(self.pc, self.pc)
                    temp_2 = temp_1 + self.th_cum * (2 - self.th_cum) * self.big_c
                    self.big_c = (1 - self.cm_learn_rate) * self.big_c + self.cm_learn_rate * temp_2

            temp_1 = self.psucc - self.tgt_sr
            self.sigma *= exp(1.0 / self.ss_dmp * temp_1 / (1.0 - self.tgt_sr))
            self.big_a = numpy.linalg.cholesky(self.big_c)

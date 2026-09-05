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
from collections.abc import Callable
from typing import Any

import numpy

from deap_er import utilities as utils
from deap_er.base.typedefs import Individual
from deap_er.rng import rng

from ._mo_update import (
    _commit_parent_params,
    _copy_offspring_state,
    _decay_rejected_offspring,
    _rank_one_update,
    _select,
    _update_chosen_offspring,
)

__all__ = ["StrategyMultiObjective"]


class StrategyMultiObjective:
    """Multi-objective Covariance Matrix Adaptation evolution strategy.

    Args:
        population: Initial parent population.
        sigma: Initial step size for every parent.
        **kwargs: Optional strategy parameters. See the table below.

    .. dropdown:: Table of Kwargs
       :margin: 0 5 0 0

       * offsprings - *(int)*
          * The number of children to produce at each generation.
          * *Default:* ``1``
       * survivors - *(int)*
          * The number of parents to keep for the next generation.
          * *Default:* ``len(population)``
       * ss_dmp - *(float)*
          * Damping of the step-size.
          * *Default:* ``1.0 + len(population[0]) / 2.0``
       * th_cum - *(float)*
          * Time horizon of the cumulative contribution.
          * *Default:* ``2.0 / (len(population[0]) + 2.0)``
       * tgt_sr - *(float)*
          * Target success rate.
          * *Default:* ``1.0 / 5.5``
       * thresh_sr - *(float)*
          * Threshold success rate.
          * *Default:* ``0.44``
       * ss_learn_rate - *(float)*
          * Learning rate of the step-size.
          * *Default:* ``tgt_sr / (2.0 + tgt_sr)``
       * cm_learn_rate - *(float)*
          * Learning rate of the covariance matrix.
          * *Default:* ``2.0 / (len(population[0]) ** 2 + 6.0)``
    """

    def __init__(self, population: list[Individual], sigma: float, **kwargs: Any) -> None:
        """See the class docstring."""
        self.parents = population
        self.dim = len(self.parents[0])
        pop_size = len(population)

        self.mu: int
        self.lamb: int
        self.ss_dmp: float
        self.tgt_sr: float
        self.ss_learn_rate: float
        self.th_cum: float
        self.cm_learn_rate: float
        self.thresh_sr: float

        self.compute_params(**kwargs)

        self.sigmas = [sigma] * pop_size
        self.big_a = [numpy.identity(self.dim) for _ in range(pop_size)]
        self.inv_cholesky = [numpy.identity(self.dim) for _ in range(pop_size)]
        self.pc = [numpy.zeros(self.dim) for _ in range(pop_size)]
        self.psucc = [self.tgt_sr] * pop_size

    _update_chosen_offspring = _update_chosen_offspring
    _rank_one_update = staticmethod(_rank_one_update)

    def compute_params(self, **kwargs: Any) -> None:
        """Recompute strategy parameters from ``kwargs``.

        Called from the constructor. Call again if ``offsprings`` or
        ``survivors`` changes during evolution.

        Args:
            **kwargs: Optional strategy parameters. See the class
                docstring.
        """
        self.mu = kwargs.get("survivors", len(self.parents))
        self.lamb = kwargs.get("offsprings", 1)
        self.ss_dmp = kwargs.get("ss_dmp", 1.0 + self.dim / 2.0)
        self.tgt_sr = kwargs.get("tgt_sr", 1.0 / (5.0 + 0.5))
        self.ss_learn_rate = kwargs.get("ss_learn_rate", self.tgt_sr / (2.0 + self.tgt_sr))
        self.th_cum = kwargs.get("th_cum", 2.0 / (self.dim + 2.0))
        self.cm_learn_rate = kwargs.get("cm_learn_rate", 2.0 / (self.dim**2 + 6.0))
        self.thresh_sr = kwargs.get("thresh_sr", 0.44)

    def update(self, population: list[Individual]) -> None:
        """Select new parents and update each parent's CMA parameters.

        Offspring are merged with the current parents, then reduced to
        ``survivors`` by non-dominated sorting. Step-size and
        covariance are updated per successful parent.

        Args:
            population: Evaluated individuals from ``generate``.
        """
        chosen, not_chosen = _select(self, population + self.parents)
        last_steps, sigmas, inv_cholesky, big_a, pc, psucc = _copy_offspring_state(self, chosen)
        _update_chosen_offspring(self, chosen, last_steps, sigmas, inv_cholesky, big_a, pc, psucc)
        _decay_rejected_offspring(self, not_chosen)
        _commit_parent_params(self, chosen, sigmas, inv_cholesky, big_a, pc, psucc)
        self.parents = chosen

    def generate(self, ind_init: Callable[..., Individual]) -> list[Individual]:
        """Sample ``offsprings`` individuals from the current parents.

        When ``offsprings`` equals the parent count, each parent
        produces one child. Otherwise parents are drawn from the first
        non-dominated front.

        Args:
            ind_init: Callable that turns a sampled vector into an
                individual.

        Returns:
            Newly sampled individuals.
        """
        arz = rng.standard_normal((self.lamb, self.dim))
        individuals = []

        for i, p in enumerate(self.parents):
            p.ps_ = "p", i

        if self.lamb == self.mu:
            for i in range(self.lamb):
                dot = numpy.dot(self.big_a[i], arz[i])
                init = ind_init(self.parents[i] + self.sigmas[i] * dot)
                individuals.append(init)
                individuals[-1].ps_ = "o", i

        else:
            n_dom = utils.sort_non_dominated(self.parents, len(self.parents))[0]

            for i in range(self.lamb):
                j = rng.integers(0, len(n_dom))
                _, p_idx = n_dom[j].ps_
                dot = numpy.dot(self.big_a[p_idx], arz[i])
                init = ind_init(self.parents[p_idx] + self.sigmas[p_idx] * dot)
                individuals.append(init)
                individuals[-1].ps_ = "o", p_idx

        return individuals

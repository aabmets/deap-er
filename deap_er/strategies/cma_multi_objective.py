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
from math import exp, sqrt
from typing import Any

import numpy

from deap_er import utilities as utils
from deap_er.base.dtypes import Individual

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
       * mp_pool - *(object)*
          * Any multiprocessing Pool object with a ``map`` method.
          * *Default:* None
    """

    def __init__(self, population: list[Individual], sigma: float, **kwargs: Any) -> None:
        """See the class docstring."""
        self.parents = population
        self.dim = len(self.parents[0])
        pop_size = len(population)

        self.mu = kwargs.get("survivors", pop_size)
        self.lamb = kwargs.get("offsprings", 1)
        self.ss_dmp = kwargs.get("ss_dmp", 1.0 + self.dim / 2.0)
        self.tgt_sr = kwargs.get("tgt_sr", 1.0 / (5.0 + 0.5))
        self.ss_learn_rate = kwargs.get("ss_learn_rate", self.tgt_sr / (2.0 + self.tgt_sr))
        self.th_cum = kwargs.get("th_cum", 2.0 / (self.dim + 2.0))
        self.cm_learn_rate = kwargs.get("cm_learn_rate", 2.0 / (self.dim**2 + 6.0))
        self.thresh_sr = kwargs.get("thresh_sr", 0.44)
        self.mp_pool = kwargs.get("mp_pool")

        self.sigmas = [sigma] * pop_size
        self.big_a = [numpy.identity(self.dim) for _ in range(pop_size)]
        self.inv_cholesky = [numpy.identity(self.dim) for _ in range(pop_size)]
        self.pc = [numpy.zeros(self.dim) for _ in range(pop_size)]
        self.psucc = [self.tgt_sr] * pop_size

    def _select(self, candidates: list[Individual]) -> tuple[list[Individual], list[Individual]]:
        """Split candidates into ``survivors`` chosen and the remainder.

        Uses non-dominated sorting. When a front would overflow
        ``survivors``, extra members are dropped by least hypervolume
        contribution.

        Args:
            candidates: Individuals to rank.

        Returns:
            Chosen individuals and those not selected.
        """
        if len(candidates) <= self.mu:
            return candidates, []

        pareto_fronts = utils.sort_log_non_dominated(candidates, len(candidates))

        chosen: list[Individual] = []
        mid_front: list[Individual] = []
        not_chosen: list[Individual] = []

        full = False
        for front in pareto_fronts:
            if len(chosen) + len(front) <= self.mu and not full:
                chosen += front
            elif not mid_front and len(chosen) < self.mu:
                mid_front = front
                full = True
            else:
                not_chosen += front

        k = self.mu - len(chosen)
        if k > 0 and mid_front:
            ref = (
                numpy.max(numpy.array([ind.fitness.wvalues for ind in candidates]) * -1, axis=0) + 1
            )

            mapper = map
            if self.mp_pool and hasattr(self.mp_pool, "map") and callable(self.mp_pool.map):
                mapper = self.mp_pool.map

            for _ in range(len(mid_front) - k):
                idx = utils.least_contrib(mid_front, ref, mapper)
                not_chosen.append(mid_front.pop(idx))

            chosen += mid_front

        return chosen, not_chosen

    @staticmethod
    def _rank_one_update(
        inv_cholesky: numpy.ndarray,
        big_a: numpy.ndarray,
        alpha: float,
        beta: float,
        v: numpy.ndarray,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Apply a rank-one covariance update to the Cholesky factors.

        Args:
            inv_cholesky: Inverse of the current Cholesky factor.
            big_a: Current Cholesky factor of the covariance.
            alpha: Weight of the existing covariance.
            beta: Weight of the rank-one term.
            v: Evolution-path vector used in the update.

        Returns:
            Updated inverse Cholesky factor and Cholesky factor.
        """
        w = numpy.dot(inv_cholesky, v)

        if w.max(initial=None) > 1e-20:
            w_inv = numpy.dot(w, inv_cholesky)
            norm_w2 = numpy.sum(w**2)
            a = sqrt(alpha)
            root = numpy.sqrt(1 + beta / alpha * norm_w2)
            b = a / norm_w2 * (root - 1)
            big_a = a * big_a + b * numpy.outer(v, w)
            part = a**2 + a * b * norm_w2
            inv_cholesky = 1.0 / a * inv_cholesky - b / part * numpy.outer(w, w_inv)

        return inv_cholesky, big_a

    def update(self, population: list[Individual]) -> None:
        """Select new parents and update each parent's CMA parameters.

        Offspring are merged with the current parents, then reduced to
        ``survivors`` by non-dominated sorting. Step-size and
        covariance are updated per successful parent.

        Args:
            population: Evaluated individuals from ``generate``.
        """
        chosen, not_chosen = self._select(population + self.parents)

        cp, cc, c_cov = self.ss_learn_rate, self.th_cum, self.cm_learn_rate
        d, pt_arg, p_thresh = self.ss_dmp, self.tgt_sr, self.thresh_sr

        bag = [list() for _ in range(6)]
        for ind in chosen:
            if ind.ps_[0] == "o":
                idx = ind.ps_[1]
                bag[0].append(self.sigmas[idx])
                bag[1].append(self.sigmas[idx])
                bag[2].append(self.inv_cholesky[idx].copy())
                bag[3].append(self.big_a[idx].copy())
                bag[4].append(self.pc[idx].copy())
                bag[5].append(self.psucc[idx])
            else:
                for b in bag:
                    b.append(None)

        last_steps, sigmas, inv_cholesky = bag[0], bag[1], bag[2]
        big_a, pc, psucc = bag[3], bag[4], bag[5]

        for i, ind in enumerate(chosen):
            t, p_idx = ind.ps_

            if t == "o":
                psucc[i] = (1.0 - cp) * psucc[i] + cp
                sigmas[i] = sigmas[i] * exp((psucc[i] - pt_arg) / (d * (1.0 - pt_arg)))

                if psucc[i] < p_thresh:
                    xp = numpy.array(ind)
                    x = numpy.array(self.parents[p_idx])
                    pc[i] = (1.0 - cc) * pc[i] + sqrt(cc * (2.0 - cc)) * (xp - x) / last_steps[i]
                    inv_cholesky[i], big_a[i] = self._rank_one_update(
                        inv_cholesky[i], big_a[i], 1 - c_cov, c_cov, pc[i]
                    )
                else:
                    pc[i] = (1.0 - cc) * pc[i]
                    pc_weight = cc * (2.0 - cc)
                    inv_cholesky[i], big_a[i] = self._rank_one_update(
                        inv_cholesky[i], big_a[i], 1 - c_cov + pc_weight, c_cov, pc[i]
                    )

                self.psucc[p_idx] = (1.0 - cp) * self.psucc[p_idx] + cp
                exp_ = exp((self.psucc[p_idx] - pt_arg) / (d * (1.0 - pt_arg)))
                self.sigmas[p_idx] = self.sigmas[p_idx] * exp_

        for ind in not_chosen:
            t, p_idx = ind.ps_

            if t == "o":
                self.psucc[p_idx] = (1.0 - cp) * self.psucc[p_idx]
                exp_ = exp((self.psucc[p_idx] - pt_arg) / (d * (1.0 - pt_arg)))
                self.sigmas[p_idx] = self.sigmas[p_idx] * exp_

        sources = {
            "inv_cholesky": inv_cholesky,
            "sigmas": sigmas,
            "big_a": big_a,
            "psucc": psucc,
            "pc": pc,
        }
        for name, var in sources.items():
            attr = getattr(self, name)
            bag = list()
            for i, ind in enumerate(chosen):
                if ind.ps_[0] == "o":
                    bag.append(var[i])
                else:
                    bag.append(attr[ind.ps_[1]])
            setattr(self, name, bag)

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
        arz = numpy.random.randn(self.lamb, self.dim)
        individuals = list()

        for i, p in enumerate(self.parents):
            p.ps_ = "p", i

        if self.lamb == self.mu:
            for i in range(self.lamb):
                dot = numpy.dot(self.big_a[i], arz[i])
                init = ind_init(self.parents[i] + self.sigmas[i] * dot)
                individuals.append(init)
                individuals[-1].ps_ = "o", i

        else:
            n_dom = utils.sort_log_non_dominated(self.parents, len(self.parents), ffo=True)

            for i in range(self.lamb):
                j = numpy.random.randint(0, len(n_dom))
                _, p_idx = n_dom[j].ps_
                dot = numpy.dot(self.big_a[p_idx], arz[i])
                init = ind_init(self.parents[p_idx] + self.sigmas[p_idx] * dot)
                individuals.append(init)
                individuals[-1].ps_ = "o", p_idx

        return individuals

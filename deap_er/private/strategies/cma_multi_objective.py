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
from typing import TYPE_CHECKING, Any

import numpy

from deap_er.private.operators.bounds import broadcast_param
from deap_er.private.various.rng import rng
from deap_er.private.various.sort_non_dominated import sort_non_dominated

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

from .common import apply_box_bounds, update_bound_attrs
from .mo_update import (
    commit_parent_params,
    copy_offspring_state,
    decay_rejected_offspring,
    rank_one_update,
    select,
    update_chosen_offspring,
)

__all__ = ["StrategyMultiObjective"]


def _raw(parent: Individual, sigma: float, big_a: numpy.ndarray, step: numpy.ndarray):
    return numpy.asarray(parent + sigma * numpy.dot(big_a, step), dtype=float)


def _front(parents: list[Individual]) -> list[Individual]:
    if all(ind.fitness.is_valid() for ind in parents):
        return sort_non_dominated(parents, len(parents))[0]
    return parents


def _resample_offspring(
    strategy: Any, ind_init: Callable[..., Individual], arz: numpy.ndarray, one_each: bool
) -> list[Individual]:
    n_dom: list[Individual] = [] if one_each else _front(strategy.parents)
    individuals = []
    for i in range(strategy.lamb):
        p_idx = i if one_each else n_dom[rng.integers(0, len(n_dom))].ps_[1]
        raw = _raw(strategy.parents[p_idx], strategy.sigmas[p_idx], strategy.big_a[p_idx], arz[i])
        init = ind_init(
            apply_box_bounds(
                raw,
                strategy.low,
                strategy.up,
                "resample",
                strategy.resample_limit,
                lambda p_idx=p_idx: _raw(
                    strategy.parents[p_idx],
                    strategy.sigmas[p_idx],
                    strategy.big_a[p_idx],
                    rng.standard_normal(strategy.dim),
                ),
            )
        )
        init.ps_ = "o", p_idx
        individuals.append(init)
    return individuals


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
       * low, up - *(float or sequence)*
          * Optional box bounds on generated individuals.
       * bound_mode - *(str)*
          * ``clip`` (default) or ``resample``. Both are
            constraint-handling approximations; the update treats the
            repaired point as the sample.
       * resample_limit - *(int)*
          * Failed redraws before clipping one sample. *Default:* ``100``
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
        self.low: Any
        self.up: Any
        self.bound_mode: str
        self.resample_limit: int

        self.compute_params(**kwargs)

        self.sigmas = [sigma] * pop_size
        self.big_a = [numpy.identity(self.dim) for _ in range(pop_size)]
        self.inv_cholesky = [numpy.identity(self.dim) for _ in range(pop_size)]
        self.pc = [numpy.zeros(self.dim) for _ in range(pop_size)]
        self.psucc = [self.tgt_sr] * pop_size

    update_chosen_offspring = update_chosen_offspring
    rank_one_update = staticmethod(rank_one_update)

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
        update_bound_attrs(self, kwargs)

    def reset_state(
        self,
        parents: list[Individual],
        sigma: float,
        **kwargs: Any,
    ) -> None:
        """Reset mutable CMA state for a restart."""
        self.compute_params(**kwargs)
        self.parents = parents[: self.mu]
        pop_size = len(self.parents)
        self.sigmas = [sigma] * pop_size
        self.big_a = [numpy.identity(self.dim) for _ in range(pop_size)]
        self.inv_cholesky = [numpy.identity(self.dim) for _ in range(pop_size)]
        self.pc = [numpy.zeros(self.dim) for _ in range(pop_size)]
        self.psucc = [self.tgt_sr] * pop_size

    def update(self, population: list[Individual]) -> None:
        """Select new parents and update each parent's CMA parameters.

        Offspring are merged with the current parents, then reduced to
        ``survivors`` by non-dominated sorting of candidates with
        valid fitness. Step-size and covariance are updated per
        successful parent.

        Args:
            population: Evaluated individuals from ``generate``.
        """
        candidates = [ind for ind in population + self.parents if ind.fitness.is_valid()]
        chosen, not_chosen = select(self, candidates)
        last_steps, sigmas, inv_cholesky, big_a, pc, psucc = copy_offspring_state(self, chosen)
        update_chosen_offspring(self, chosen, last_steps, sigmas, inv_cholesky, big_a, pc, psucc)
        decay_rejected_offspring(self, not_chosen, chosen)
        commit_parent_params(self, chosen, sigmas, inv_cholesky, big_a, pc, psucc)
        self.parents = chosen

    def generate(self, ind_init: Callable[..., Individual]) -> list[Individual]:
        """Sample ``offsprings`` individuals from the current parents.

        When ``offsprings`` equals the parent count and that many
        parents exist, each parent produces one child. Otherwise
        parents are drawn from the first non-dominated front, or from
        every parent if any parent fitness is invalid.

        Args:
            ind_init: Callable that turns a sampled vector into an
                individual.

        Returns:
            Newly sampled individuals.
        """
        arz = rng.standard_normal((self.lamb, self.dim))
        for i, p in enumerate(self.parents):
            p.ps_ = "p", i
        if not self.parents:
            return []
        one_each = self.lamb == self.mu and len(self.parents) >= self.lamb
        if self.bound_mode == "resample" and (self.low is not None or self.up is not None):
            return _resample_offspring(self, ind_init, arz, one_each)
        if one_each:
            parent_idxs = list(range(self.lamb))
        else:
            n_dom = _front(self.parents)
            parent_idxs = [n_dom[rng.integers(0, len(n_dom))].ps_[1] for _ in range(self.lamb)]
        raws = numpy.empty((self.lamb, self.dim), dtype=float)
        for i, p_idx in enumerate(parent_idxs):
            raws[i] = _raw(self.parents[p_idx], self.sigmas[p_idx], self.big_a[p_idx], arz[i])
        if self.low is not None or self.up is not None:
            low_seq = broadcast_param("low", -numpy.inf if self.low is None else self.low, self.dim)
            up_seq = broadcast_param("up", numpy.inf if self.up is None else self.up, self.dim)
            raws = numpy.clip(
                raws, numpy.asarray(low_seq, dtype=float), numpy.asarray(up_seq, dtype=float)
            )
        individuals = []
        for i, p_idx in enumerate(parent_idxs):
            init = ind_init(raws[i])
            init.ps_ = "o", p_idx
            individuals.append(init)
        return individuals

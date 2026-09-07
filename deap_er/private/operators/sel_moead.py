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

from itertools import chain
from operator import attrgetter
from typing import TYPE_CHECKING, Literal

import numpy
from numpy import ndarray

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual
from deap_er.private.various.sort_non_dominated import sort_non_dominated

from .sel_helpers import assign_crowding_dist
from .sel_moead_helpers import (
    ScalarizationFn,
    scalarization_pbi,
    scalarization_tchebycheff,
)

__all__: list[str] = ["sel_moead", "SelMOEADWithMemory"]

ScalarizationName = Literal["tchebycheff", "pbi"]


def _resolve_scalarization(
    scalarization: ScalarizationName | ScalarizationFn,
    theta: float,
) -> ScalarizationFn:
    if scalarization == "tchebycheff":
        return scalarization_tchebycheff
    if scalarization == "pbi":
        return lambda fitness, weights, ideal: scalarization_pbi(
            fitness, weights, ideal, theta=theta
        )
    return scalarization


def _minimize_fitness(individuals: list[Individual]) -> ndarray:
    return -numpy.array([ind.fitness.wvalues for ind in individuals], dtype=float)


def _update_ideal_point(fitness: ndarray, ideal_point: ndarray | None) -> ndarray:
    current = numpy.min(fitness, axis=0)
    if ideal_point is None or ideal_point.size == 0:
        return current
    finite = ideal_point[numpy.isfinite(ideal_point)]
    if finite.size == 0:
        return current
    return numpy.minimum(current, finite)


def _pareto_ranks(individuals: list[Individual]) -> ndarray:
    fronts = sort_non_dominated(individuals, len(individuals))
    ranks = numpy.full(len(individuals), len(individuals), dtype=numpy.int64)
    index_map = {id(ind): idx for idx, ind in enumerate(individuals)}
    for rank, front in enumerate(fronts):
        for ind in front:
            ranks[index_map[id(ind)]] = rank
    return ranks


def _select_by_subproblems(
    individuals: list[Individual],
    fitness: ndarray,
    weights: ndarray,
    ideal_point: ndarray,
    scalar_fn: ScalarizationFn,
    sel_count: int,
    ranks: ndarray,
) -> list[Individual]:
    n_weights = min(sel_count, len(weights))
    scalar = scalar_fn(fitness, weights[:n_weights], ideal_point)
    chosen: list[Individual] = []
    used = numpy.zeros(len(individuals), dtype=numpy.bool)
    for niche in range(n_weights):
        order = numpy.lexsort((scalar[:, niche], ranks))
        for idx in order:
            if not used[idx]:
                used[idx] = True
                chosen.append(individuals[idx])
                break
    return chosen


def _fill_with_crowding(
    individuals: list[Individual],
    chosen: list[Individual],
    sel_count: int,
) -> list[Individual]:
    if len(chosen) >= sel_count:
        return chosen[:sel_count]
    need = sel_count - len(chosen)
    chosen_ids = {id(ind) for ind in chosen}
    pool = [ind for ind in individuals if id(ind) not in chosen_ids]
    if not pool:
        return chosen

    fronts = sort_non_dominated(pool, need)
    for front in fronts:
        assign_crowding_dist(front)

    extra = list(chain(*fronts[:-1]))
    still_need = need - len(extra)
    if still_need > 0 and fronts:
        attr = attrgetter("fitness.crowding_dist")
        sorted_last = sorted(fronts[-1], key=attr, reverse=True)
        extra.extend(sorted_last[:still_need])
    chosen.extend(extra)
    return chosen[:sel_count]


class SelMOEADWithMemory:
    """MOEA/D selection that remembers the ideal point across generations.

    Instances can be registered into a Toolbox.

    Args:
        weights: Decomposition weight vectors.
        scalarization: ``"tchebycheff"``, ``"pbi"``, or a callable.
        theta: PBI penalty parameter.
    """

    def __init__(
        self,
        weights: ndarray,
        *,
        scalarization: ScalarizationName | ScalarizationFn = "tchebycheff",
        theta: float = 5.0,
    ) -> None:
        """See the class docstring."""
        self.weights = weights
        self.scalarization = scalarization
        self.theta = theta
        self.ideal_point = numpy.full((1, weights.shape[1]), numpy.inf)

    def __call__(self, individuals: list[Individual], sel_count: int) -> list[Individual]:
        """Select individuals for the next generation.

        Args:
            individuals: Individuals to select from.
            sel_count: Number of individuals to select.

        Returns:
            The selected individuals.
        """
        return sel_moead(
            individuals,
            sel_count,
            self.weights,
            scalarization=self.scalarization,
            theta=self.theta,
            ideal_point=self.ideal_point,
            _memory=self,
        )


def sel_moead(
    individuals: list[Individual],
    sel_count: int,
    weights: ndarray,
    *,
    scalarization: ScalarizationName | ScalarizationFn = "tchebycheff",
    theta: float = 5.0,
    ideal_point: ndarray | None = None,
    _memory: SelMOEADWithMemory | None = None,
) -> list[Individual]:
    """Select the next generation with MOEA/D decomposition.

    Each weight vector defines a scalar subproblem. ``weights`` is
    typically ``uniform_reference_points``. Subproblem winners
    prefer lower Pareto ranks; remaining slots are filled from
    complete lower fronts, then crowding distance on the last
    partial front of the leftover pool.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.
        weights: Decomposition weight vectors with shape ``(k, m)``.
        scalarization: ``"tchebycheff"``, ``"pbi"``, or a callable
            ``(fitness, weights, ideal_point) -> ndarray``.
        theta: PBI penalty parameter.
        ideal_point: Ideal point from a previous generation. If
            omitted, it is taken from the current individuals.
        _memory: ``SelMOEADWithMemory`` instance that stores the
            updated ideal point. Not intended for manual use.

    Returns:
        The selected individuals.
    """
    if not individuals or sel_count <= 0:
        return []
    if sel_count >= len(individuals):
        return list(individuals)

    fitness = _minimize_fitness(individuals)
    scalar_fn = _resolve_scalarization(scalarization, theta)
    ranks = _pareto_ranks(individuals)

    prior = None
    if ideal_point is not None:
        prior = numpy.asarray(ideal_point, dtype=float).reshape(-1)
        prior = prior[numpy.isfinite(prior)]
    elif isinstance(_memory, SelMOEADWithMemory):
        prior = numpy.asarray(_memory.ideal_point, dtype=float).reshape(-1)
        prior = prior[numpy.isfinite(prior)]

    z_star = _update_ideal_point(fitness, prior)
    chosen = _select_by_subproblems(
        individuals, fitness, weights, z_star, scalar_fn, sel_count, ranks
    )
    chosen = _fill_with_crowding(individuals, chosen, sel_count)

    if isinstance(_memory, SelMOEADWithMemory):
        _memory.ideal_point = numpy.asarray(z_star).reshape((1, -1))

    return chosen

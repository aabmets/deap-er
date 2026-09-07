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
from typing import TYPE_CHECKING

import numpy
from numpy import ndarray

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual
from deap_er.private.various.sort_non_dominated import sort_non_dominated

from .sel_age_moea_2_helpers import estimate_curvature_nr, survival_scores
from .sel_nsga_3_helpers import find_extreme_points, find_intercepts

__all__: list[str] = ["sel_age_moea_2", "SelAGE2WithMemory"]


def _normalize_front(
    fitness: ndarray,
    best_point: ndarray,
    worst_point: ndarray,
    extreme_points: ndarray | None,
) -> tuple[ndarray, ndarray, ndarray]:
    best = numpy.min(fitness, axis=0) if best_point is None else best_point
    worst = numpy.max(fitness, axis=0) if worst_point is None else worst_point
    extreme = find_extreme_points(fitness, best, extreme_points)
    intercepts = find_intercepts(extreme, best, worst, worst)
    denom = intercepts - best
    denom = numpy.where(numpy.abs(denom) < 1e-12, 1.0, denom)
    normalized = (fitness - best) / denom
    return normalized, best, worst


def find_extreme_indexes(fitness: ndarray, best_point: ndarray) -> ndarray:
    """Return one extreme-point index per objective.

    Args:
        fitness: Objective matrix with shape ``(n, m)``.
        best_point: Ideal point with shape ``(m,)``.

    Returns:
        Integer indexes into ``fitness``.
    """
    ft = fitness - best_point
    asf = numpy.eye(best_point.shape[0])
    asf[asf == 0] = 1e6
    asf = numpy.max(ft * asf[:, numpy.newaxis, :], axis=2)
    return numpy.argmin(asf, axis=1)


def _reference_index(front: ndarray, extreme_indexes: ndarray) -> int:
    distances = numpy.linalg.norm(front, axis=1)
    distances[extreme_indexes] = numpy.inf
    return int(numpy.argmin(distances))


def _geometry_from_first_front(
    first_front: ndarray,
    best_point: ndarray,
    worst_point: ndarray,
    extreme_points: ndarray | None,
    nr_tol: float,
    nr_max_iter: int,
) -> tuple[float, ndarray, ndarray]:
    normalized, best, worst = _normalize_front(first_front, best_point, worst_point, extreme_points)
    extreme_idx = find_extreme_indexes(first_front, best)
    ref = _reference_index(normalized, extreme_idx)
    curvature = estimate_curvature_nr(normalized[ref], normalized.shape[1], nr_tol, nr_max_iter)
    return curvature, best, worst


class SelAGE2WithMemory:
    """AGE-MOEA-II selection that remembers normalization anchors.

    Instances can be registered into a Toolbox.
    """

    def __init__(self) -> None:
        """See the class docstring."""
        self.best_point = numpy.array([])
        self.worst_point = numpy.array([])
        self.extreme_points: ndarray | None = None
        self.curvature = 1.0

    def __call__(self, individuals: list[Individual], sel_count: int) -> list[Individual]:
        """Select individuals for the next generation.

        Args:
            individuals: Individuals to select from.
            sel_count: Number of individuals to select.

        Returns:
            The selected individuals.
        """
        best = self.best_point.reshape(-1) if self.best_point.size else None
        worst = self.worst_point.reshape(-1) if self.worst_point.size else None
        return sel_age_moea_2(
            individuals,
            sel_count,
            best_point=best,
            worst_point=worst,
            extreme_points=self.extreme_points,
            _memory=self,
        )


def sel_age_moea_2(
    individuals: list[Individual],
    sel_count: int,
    *,
    best_point: ndarray | None = None,
    worst_point: ndarray | None = None,
    extreme_points: ndarray | None = None,
    nr_tol: float = 1e-3,
    nr_max_iter: int = 100,
    _memory: SelAGE2WithMemory | None = None,
) -> list[Individual]:
    """Select the next generation with AGE-MOEA-II.

    Non-dominated sorting fills complete fronts first. Remaining
    slots on the last partial front use geometry-aware survival
    scores based on Newton-Raphson curvature estimation and
    geodesic diversity.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.
        best_point: Ideal point of the previous generation. If
            omitted, it is taken from the current individuals.
        worst_point: Nadir point of the previous generation. If
            omitted, it is taken from the current individuals.
        extreme_points: Extreme points of the previous generation.
            If omitted, they are taken from the current individuals.
        nr_tol: Newton-Raphson stopping tolerance for curvature.
        nr_max_iter: Maximum Newton-Raphson iterations.
        _memory: ``SelAGE2WithMemory`` instance that stores the
            updated normalization anchors. Not intended for manual use.

    Returns:
        The selected individuals.
    """
    if not individuals or sel_count <= 0:
        return []
    if sel_count >= len(individuals):
        return list(individuals)

    pareto_fronts = sort_non_dominated(individuals, sel_count)
    fitness = -numpy.array([ind.fitness.wvalues for ind in individuals], dtype=float)
    index_map = {id(ind): idx for idx, ind in enumerate(individuals)}

    if best_point is not None and worst_point is not None:
        best = numpy.min(numpy.vstack((fitness, best_point.reshape(1, -1))), axis=0)
        worst = numpy.max(numpy.vstack((fitness, worst_point.reshape(1, -1))), axis=0)
    else:
        best = numpy.min(fitness, axis=0)
        worst = numpy.max(fitness, axis=0)

    first_indices = [index_map[id(ind)] for ind in pareto_fronts[0]]
    first_front = fitness[first_indices]
    curvature, best, worst = _geometry_from_first_front(
        first_front,
        best,
        worst,
        extreme_points,
        nr_tol,
        nr_max_iter,
    )

    chosen = list(chain(*pareto_fronts[:-1]))
    remaining = sel_count - len(chosen)
    if remaining <= 0:
        return chosen[:sel_count]

    last_front = pareto_fronts[-1]
    last_indices = numpy.array([index_map[id(ind)] for ind in last_front], dtype=int)
    last_fitness = fitness[last_indices]
    normalized, _, _ = _normalize_front(last_fitness, best, worst, None)
    extreme_last = find_extreme_indexes(last_fitness, best)
    scores = survival_scores(normalized, numpy.zeros(normalized.shape[1]), extreme_last, curvature)
    order = numpy.argsort(scores)[::-1]
    chosen.extend(last_front[idx] for idx in order[:remaining])

    if isinstance(_memory, SelAGE2WithMemory):
        _memory.best_point = numpy.asarray(best).reshape((1, -1))
        _memory.worst_point = numpy.asarray(worst).reshape((1, -1))
        _memory.extreme_points = find_extreme_points(first_front, best, extreme_points)
        _memory.curvature = curvature

    return chosen

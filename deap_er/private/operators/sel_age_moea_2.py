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

from typing import TYPE_CHECKING

import numpy
from numpy import ndarray

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual
from deap_er.private.various.sort_non_dominated import sort_non_dominated

from .sel_age_moea_2_helpers import (
    estimate_curvature_nr,
    later_front_scores,
    survival_scores,
)
from .sel_nsga_3_helpers import find_extreme_points, find_intercepts

__all__: list[str] = ["sel_age_moea_2", "SelAGE2WithMemory"]


def _normalize_front(
    fitness: ndarray,
    best_point: ndarray,
    worst_point: ndarray,
    extreme_points: ndarray | None,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    best = best_point
    worst = worst_point
    extreme = find_extreme_points(fitness, best, extreme_points)
    intercepts = find_intercepts(extreme, best, worst, worst)
    denom = intercepts - best
    denom = numpy.where(numpy.abs(denom) < 1e-12, 1.0, denom)
    normalized = (fitness - best) / denom
    return normalized, best, worst, intercepts


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
    if not numpy.any(numpy.isfinite(distances)):
        return 0
    return int(numpy.argmin(distances))


def _merge_best(fitness: ndarray, anchor: ndarray | None) -> ndarray:
    if anchor is None:
        return numpy.min(fitness, axis=0)
    row = numpy.asarray(anchor, dtype=float).reshape(1, -1)
    return numpy.min(numpy.vstack((fitness, row)), axis=0)


def _merge_worst(fitness: ndarray, anchor: ndarray | None) -> ndarray:
    if anchor is None:
        return numpy.max(fitness, axis=0)
    row = numpy.asarray(anchor, dtype=float).reshape(1, -1)
    return numpy.max(numpy.vstack((fitness, row)), axis=0)


def _estimate_geometry(
    first_front: ndarray,
    best: ndarray,
    worst: ndarray,
    extreme_points: ndarray | None,
    nr_tol: float,
    nr_max_iter: int,
) -> tuple[float, ndarray]:
    normalized, _, _, intercepts = _normalize_front(first_front, best, worst, extreme_points)
    extreme_idx = find_extreme_indexes(first_front, best)
    ref = _reference_index(normalized, extreme_idx)
    curvature = estimate_curvature_nr(normalized[ref], normalized.shape[1], nr_tol, nr_max_iter)
    return curvature, intercepts


def _front_survival_scores(
    front_fitness: ndarray,
    front_index: int,
    best: ndarray,
    worst: ndarray,
    intercepts: ndarray,
    curvature: float,
    extreme_points: ndarray | None,
) -> ndarray:
    if front_index == 0:
        normalized, _, _, _ = _normalize_front(front_fitness, best, worst, extreme_points)
        extreme_idx = find_extreme_indexes(front_fitness, best)
        return survival_scores(normalized, numpy.zeros(normalized.shape[1]), extreme_idx, curvature)
    return later_front_scores(front_fitness, best, intercepts, curvature)


def _update_memory(
    memory: SelAGE2WithMemory,
    best: ndarray,
    worst: ndarray,
    first_front: ndarray,
    extreme_points: ndarray | None,
    curvature: float,
) -> None:
    memory.best_point = numpy.asarray(best).reshape((1, -1))
    memory.worst_point = numpy.asarray(worst).reshape((1, -1))
    memory.extreme_points = find_extreme_points(first_front, best, extreme_points)
    memory.curvature = curvature


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

    Follows the environmental-selection loop of Panichella (GECCO
    2022, Algorithm 2): non-dominated fronts are processed in order;
    when a front does not fit entirely, survivors are chosen by
    geometry-aware scores. The first front uses Newton-Raphson
    curvature and geodesic diversity; later fronts reuse the first
    front normalization and rank by inverse Minkowski distance to the
    ideal point.

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
        if isinstance(_memory, SelAGE2WithMemory):
            fitness = -numpy.array([ind.fitness.wvalues for ind in individuals], dtype=float)
            best = _merge_best(fitness, best_point)
            worst = _merge_worst(fitness, worst_point)
            first_front = fitness
            curvature, _ = _estimate_geometry(
                first_front, best, worst, extreme_points, nr_tol, nr_max_iter
            )
            _update_memory(_memory, best, worst, first_front, extreme_points, curvature)
        return list(individuals)

    pareto_fronts = sort_non_dominated(individuals, sel_count)
    fitness = -numpy.array([ind.fitness.wvalues for ind in individuals], dtype=float)
    index_map = {id(ind): idx for idx, ind in enumerate(individuals)}

    best = _merge_best(fitness, best_point)
    worst = _merge_worst(fitness, worst_point)

    first_indices = [index_map[id(ind)] for ind in pareto_fronts[0]]
    first_front = fitness[first_indices]
    curvature, intercepts = _estimate_geometry(
        first_front, best, worst, extreme_points, nr_tol, nr_max_iter
    )

    chosen: list[Individual] = []
    for front_index, front in enumerate(pareto_fronts):
        if len(chosen) >= sel_count:
            break
        if len(chosen) + len(front) <= sel_count:
            chosen.extend(front)
            continue

        remaining = sel_count - len(chosen)
        front_indices = numpy.array([index_map[id(ind)] for ind in front], dtype=int)
        front_fitness = fitness[front_indices]
        scores = _front_survival_scores(
            front_fitness,
            front_index,
            best,
            worst,
            intercepts,
            curvature,
            extreme_points,
        )
        order = numpy.argsort(scores)[::-1]
        chosen.extend(front[idx] for idx in order[:remaining])

    if isinstance(_memory, SelAGE2WithMemory):
        _update_memory(_memory, best, worst, first_front, extreme_points, curvature)

    return chosen[:sel_count]

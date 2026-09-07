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

from .sel_age_moea_2_helpers import estimate_curvature_nr
from .sel_nsga_3_helpers import find_extreme_points, find_intercepts

__all__: list[str] = [
    "merge_best",
    "merge_worst",
    "normalize_front",
    "pareto_front_fitness",
    "estimate_geometry",
    "extreme_indexes",
]


def normalize_front(
    fitness: ndarray,
    best_point: ndarray,
    worst_point: ndarray,
    extreme_points: ndarray | None,
    *,
    front_worst: ndarray | None = None,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """Normalize ``fitness`` using NSGA-III-style intercept estimation.

    Args:
        fitness: Objective matrix with shape ``(n, m)``.
        best_point: Ideal point with shape ``(m,)``.
        worst_point: Memory nadir with shape ``(m,)``.
        extreme_points: Extreme points from a previous generation.
        front_worst: Front-local nadir for intercept fallback.

    Returns:
        Normalized objectives, ideal, nadir, and intercepts.
    """
    best = best_point
    worst = worst_point
    local_worst = front_worst if front_worst is not None else numpy.max(fitness, axis=0)
    extreme = find_extreme_points(fitness, best, extreme_points)
    intercepts = find_intercepts(extreme, best, worst, local_worst)
    denom = intercepts - best
    denom = numpy.where(numpy.abs(denom) < 1e-12, 1.0, denom)
    normalized = (fitness - best) / denom
    return normalized, best, worst, intercepts


def _find_extreme_indexes(fitness: ndarray, best_point: ndarray) -> ndarray:
    ft = fitness - best_point
    asf = numpy.eye(best_point.shape[0])
    asf[asf == 0] = 1e6
    asf = numpy.max(ft * asf[:, numpy.newaxis, :], axis=2)
    return numpy.argmin(asf, axis=1)


def extreme_indexes(fitness: ndarray, best_point: ndarray) -> ndarray:
    """Return one extreme-point index per objective on ``fitness``."""
    return _find_extreme_indexes(fitness, best_point)


def _reference_index(front: ndarray, extreme_indexes: ndarray) -> int:
    distances = numpy.linalg.norm(front, axis=1)
    distances[extreme_indexes] = numpy.inf
    if not numpy.any(numpy.isfinite(distances)):
        return 0
    return int(numpy.argmin(distances))


def pareto_front_fitness(
    fitness: ndarray,
    pareto_fronts: list[list[Individual]],
    index_map: dict[int, int],
) -> ndarray:
    """Collect minimize-space fitness for individuals in ``pareto_fronts``."""
    indices = [index_map[id(ind)] for front in pareto_fronts for ind in front]
    return fitness[numpy.asarray(indices, dtype=int)]


def merge_best(fitness: ndarray, anchor: ndarray | None) -> ndarray:
    """Merge per-objective minima from ``fitness`` and an optional anchor."""
    if anchor is None:
        return numpy.min(fitness, axis=0)
    row = numpy.asarray(anchor, dtype=float).reshape(1, -1)
    return numpy.min(numpy.vstack((fitness, row)), axis=0)


def merge_worst(fitness: ndarray, anchor: ndarray | None) -> ndarray:
    """Merge per-objective maxima from ``fitness`` and a finite anchor."""
    current = numpy.max(fitness, axis=0)
    if anchor is None:
        return current
    row = numpy.asarray(anchor, dtype=float).reshape(1, -1)
    finite = numpy.isfinite(row)
    if not numpy.any(finite):
        return current
    safe_row = numpy.where(finite, row, -numpy.inf)
    return numpy.max(numpy.vstack((fitness, safe_row)), axis=0)


def estimate_geometry(
    first_front: ndarray,
    best: ndarray,
    worst: ndarray,
    extreme_points: ndarray | None,
    nr_tol: float,
    nr_max_iter: int,
    front_worst: ndarray,
) -> tuple[float, ndarray]:
    """Estimate AGE-MOEA-II curvature and intercepts from the first front."""
    normalized, _, _, intercepts = normalize_front(
        first_front, best, worst, extreme_points, front_worst=front_worst
    )
    if first_front.shape[0] < first_front.shape[1]:
        return 1.0, intercepts
    extreme_idx = _find_extreme_indexes(first_front, best)
    ref = _reference_index(normalized, extreme_idx)
    curvature = estimate_curvature_nr(normalized[ref], normalized.shape[1], nr_tol, nr_max_iter)
    return curvature, intercepts

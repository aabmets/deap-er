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

import numpy
from numpy import ndarray

__all__: list[str] = [
    "project_on_manifold",
    "geodesic_distance",
    "geodesic_distance_matrix",
    "survival_scores",
    "later_front_scores",
]

_ZERO_TOL = 1e-12


def project_on_manifold(point: ndarray, curvature: float) -> ndarray:
    """Project a normalized point onto the unit ``L^p`` manifold.

    Args:
        point: Normalized objective vector.
        curvature: Estimated front curvature ``p``.

    Returns:
        Projected point on the unit manifold.
    """
    positive = point[point > 0.0]
    if positive.size == 0:
        return point.copy()
    dist = numpy.sum(positive**curvature) ** (1.0 / curvature)
    if abs(dist) < _ZERO_TOL:
        return point.copy()
    return point / dist


def geodesic_distance(a: ndarray, b: ndarray, curvature: float) -> float:
    """Approximate the geodesic distance between two manifold points.

    Args:
        a: First normalized objective vector.
        b: Second normalized objective vector.
        curvature: Estimated front curvature ``p``.

    Returns:
        Approximate geodesic distance.
    """
    if 0.95 < curvature < 1.05:
        return float(numpy.linalg.norm(a - b))

    mid = 0.5 * (a + b)
    mid_proj = project_on_manifold(mid, curvature)
    return float(numpy.linalg.norm(a - mid_proj) + numpy.linalg.norm(b - mid_proj))


def _geodesic_matrix_curved(projected: ndarray, curvature: float) -> ndarray:
    """Return pairwise geodesic distances for curvature outside the flat band.

    Midpoints are projected with the same ``L^p`` rule as
    ``project_on_manifold``: only positive coordinates enter the
    norm, and a near-zero norm leaves the midpoint unchanged.

    Args:
        projected: Manifold-projected points with shape ``(n, m)``.
        curvature: Estimated front curvature ``p``.

    Returns:
        Symmetric distance matrix with shape ``(n, n)``.
    """
    mid = 0.5 * (projected[:, numpy.newaxis, :] + projected[numpy.newaxis, :, :])
    pos = numpy.where(mid > 0.0, mid, 0.0)
    dist = numpy.sum(pos**curvature, axis=2) ** (1.0 / curvature)
    dist = numpy.where(numpy.abs(dist) < _ZERO_TOL, 1.0, dist)
    mid_proj = mid / dist[..., numpy.newaxis]
    distances = numpy.linalg.norm(projected[:, numpy.newaxis, :] - mid_proj, axis=2)
    distances += numpy.linalg.norm(projected[numpy.newaxis, :, :] - mid_proj, axis=2)
    numpy.fill_diagonal(distances, 0.0)
    return distances


def geodesic_distance_matrix(front: ndarray, curvature: float) -> ndarray:
    """Return pairwise geodesic distances for a normalized front.

    Args:
        front: Objective matrix with shape ``(n, m)``.
        curvature: Estimated front curvature ``p``.

    Returns:
        Symmetric distance matrix with shape ``(n, n)``.
    """
    n = front.shape[0]
    if n == 0:
        return numpy.zeros((0, 0), dtype=float)
    projected = numpy.array([project_on_manifold(row, curvature) for row in front])

    if 0.95 < curvature < 1.05:
        distances = numpy.zeros((n, n), dtype=float)
        for row in range(n - 1):
            diff = projected[row + 1 :] - projected[row]
            row_dist = numpy.linalg.norm(diff, axis=1)
            distances[row, row + 1 :] = row_dist
            distances[row + 1 :, row] = row_dist
        return distances

    return _geodesic_matrix_curved(projected, curvature)


def survival_scores(
    front: ndarray,
    ideal_point: ndarray,
    extreme: ndarray,
    curvature: float,
) -> ndarray:
    """Assign geometry-aware survival scores to a normalized front.

    Higher scores are better. Extreme points receive infinite score.

    Args:
        front: Normalized objective matrix with shape ``(n, m)``.
        ideal_point: Ideal point with shape ``(m,)``.
        extreme: Indexes of extreme points on ``front``.
        curvature: Estimated front curvature ``p``.

    Returns:
        Survival score per row of ``front``.
    """
    n = front.shape[0]
    scores = numpy.zeros(n, dtype=float)
    scores[extreme] = numpy.inf
    selected = numpy.zeros(n, dtype=numpy.bool)
    selected[extreme] = True

    distances = geodesic_distance_matrix(front, curvature)
    distances[distances < 1e-8] = 1e-8
    norms = numpy.linalg.norm(front, ord=curvature, axis=1)
    norms[norms < 1e-8] = 1.0
    distances = distances / norms[:, numpy.newaxis]

    remaining = [idx for idx in range(n) if not selected[idx]]
    neighbors = 2
    for _ in range(n - int(numpy.sum(selected))):
        if not remaining:
            break
        best_score = -1.0
        best_idx = remaining[0]
        selected_rows = numpy.flatnonzero(selected)
        for rem_idx in remaining:
            dists = distances[rem_idx, selected_rows]
            if dists.size > 1:
                part = numpy.partition(dists, neighbors - 1)[:neighbors]
                score = float(numpy.sum(part))
            else:
                score = float(dists[0])
            if score > best_score:
                best_score = score
                best_idx = rem_idx
        selected[best_idx] = True
        scores[best_idx] = best_score
        remaining.remove(best_idx)

    convergence = numpy.linalg.norm(front - ideal_point, ord=curvature, axis=1)
    scores = scores + 1.0 / (convergence + 1e-8)
    return scores


def later_front_scores(
    front: ndarray,
    best_point: ndarray,
    intercepts: ndarray,
    curvature: float,
) -> ndarray:
    """Score a non-first front by inverse Minkowski distance to the ideal.

    Later fronts reuse the normalization hyperplane estimated from the
    first non-dominated front, matching AGE-MOEA-II / pymoo behavior.

    Args:
        front: Raw objective matrix with shape ``(n, m)``.
        best_point: Ideal point with shape ``(m,)``.
        intercepts: Axis intercepts from the first front.
        curvature: Estimated front curvature ``p``.

    Returns:
        Survival score per row; higher is better.
    """
    denom = intercepts - best_point
    denom = numpy.where(numpy.abs(denom) < 1e-12, 1.0, denom)
    normalized = (front - best_point) / denom
    ideal = numpy.zeros(front.shape[1])
    dist = numpy.linalg.norm(normalized - ideal, ord=curvature, axis=1)
    dist[dist < 1e-8] = 1e-8
    return 1.0 / dist

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

import numpy
from numpy import ndarray

__all__: list[str] = [
    "scalarization_tchebycheff",
    "scalarization_pbi",
    "moead_neighborhood",
    "ScalarizationFn",
]

ScalarizationFn = Callable[[ndarray, ndarray, ndarray], ndarray]


def scalarization_tchebycheff(
    fitness: ndarray,
    weights: ndarray,
    ideal_point: ndarray,
    eps: float = 1e-16,
) -> ndarray:
    """Return Tchebycheff scalarized values for each individual and weight.

    Lower values are better. ``fitness`` and ``ideal_point`` are in
    minimize space.

    Args:
        fitness: Objective matrix with shape ``(n_ind, m)``.
        weights: Weight matrix with shape ``(k, m)``.
        ideal_point: Ideal point with shape ``(m,)``.
        eps: Small constant added to zero weights.

    Returns:
        Scalarized matrix with shape ``(n_ind, k)``.
    """
    diff = numpy.abs(fitness - ideal_point)
    w = numpy.where(numpy.abs(weights) < eps, eps, weights)
    return numpy.max(diff[:, numpy.newaxis, :] * w[numpy.newaxis, :, :], axis=2)


def scalarization_pbi(
    fitness: ndarray,
    weights: ndarray,
    ideal_point: ndarray,
    theta: float = 5.0,
    eps: float = 1e-16,
) -> ndarray:
    """Return PBI scalarized values for each individual and weight.

    Lower values are better. ``fitness`` and ``ideal_point`` are in
    minimize space.

    Args:
        fitness: Objective matrix with shape ``(n_ind, m)``.
        weights: Weight matrix with shape ``(k, m)``.
        ideal_point: Ideal point with shape ``(m,)``.
        theta: Penalty parameter balancing convergence and diversity.
        eps: Small constant added to zero weights.

    Returns:
        Scalarized matrix with shape ``(n_ind, k)``.
    """
    diff = fitness - ideal_point
    w = numpy.where(numpy.abs(weights) < eps, eps, weights)
    norm_w = numpy.linalg.norm(w, axis=1)
    d1 = numpy.sum(diff[:, numpy.newaxis, :] * w[numpy.newaxis, :, :], axis=2) / norm_w
    unit = w / norm_w[:, numpy.newaxis]
    proj = ideal_point + d1[:, :, numpy.newaxis] * unit[numpy.newaxis, :, :]
    d2 = numpy.linalg.norm(fitness[:, numpy.newaxis, :] - proj, axis=2)
    return d1 + theta * d2


def moead_neighborhood(weights: ndarray, n_neighbors: int) -> ndarray:
    """Return sorted neighbor indices for each weight vector.

    Each row lists the ``n_neighbors`` closest weight vectors by
    Euclidean distance, including the index itself.

    Args:
        weights: Weight matrix with shape ``(k, m)``.
        n_neighbors: Number of neighbors per weight vector.

    Returns:
        Integer matrix with shape ``(k, n_neighbors)``.
    """
    n_neighbors = min(n_neighbors, len(weights))
    dist = numpy.linalg.norm(weights[:, numpy.newaxis, :] - weights[numpy.newaxis, :, :], axis=2)
    return numpy.argsort(dist, axis=1, kind="quicksort")[:, :n_neighbors]

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
from collections.abc import Sequence
from typing import Any, Literal

import numpy

from deap_er.private.various.semantic_mask import (
    as_semantic_matrix,
    semantic_valid_mask,
    validate_semantic_matrix,
)

__all__: list[str] = ["semantic_distance", "semantic_nearest"]

type SemanticMetric = Literal["euclidean", "cosine"]


def _query_row(query: numpy.ndarray | Sequence[float], n_rows: int) -> numpy.ndarray:
    row = numpy.asarray(query, dtype=numpy.float64)
    if row.ndim != 1 or row.shape[0] != n_rows:
        raise ValueError(f"query must have shape ({n_rows},), got {row.shape}")
    return row


def _pair_mask(
    packed: numpy.ndarray,
    query: numpy.ndarray,
    valid: numpy.ndarray | None,
) -> numpy.ndarray:
    row_valid = semantic_valid_mask(packed, valid)
    query_ok = numpy.isfinite(query)
    if valid is not None:
        sample_valid = numpy.asarray(valid, dtype=bool)
        query_ok = query_ok & sample_valid
    return row_valid & query_ok


def _masked_distance(
    packed: numpy.ndarray,
    query: numpy.ndarray,
    pair: numpy.ndarray,
    metric: SemanticMetric,
) -> numpy.ndarray:
    if metric == "euclidean":
        diff = numpy.where(pair, packed - query, 0.0)
        dist = numpy.sqrt(numpy.square(diff).sum(axis=1))
        return numpy.where(pair.any(axis=1), dist, numpy.inf)
    if metric == "cosine":
        left = numpy.where(pair, packed, 0.0)
        right = numpy.where(pair, query, 0.0)
        dot = (left * right).sum(axis=1)
        norm_left = numpy.sqrt(numpy.square(left).sum(axis=1))
        norm_right = numpy.sqrt(numpy.square(right).sum(axis=1))
        denom = norm_left * norm_right
        sim = numpy.divide(dot, denom, out=numpy.full(dot.shape, numpy.nan), where=denom > 0.0)
        dist = 1.0 - sim
        bad = ~pair.any(axis=1) | (denom <= 0.0)
        return numpy.where(bad, numpy.inf, dist)
    raise ValueError(f"metric must be 'euclidean' or 'cosine', got {metric!r}")


def semantic_distance(
    query: numpy.ndarray | Sequence[float],
    matrix: numpy.ndarray | Sequence[Sequence[float]] | Sequence[float],
    *,
    metric: SemanticMetric = "euclidean",
    valid: numpy.ndarray | None = None,
) -> numpy.ndarray | float:
    """Return finite-mask distances from ``query`` to each packed row.

    Only coordinates that are finite on both sides and marked ``valid``
    enter the distance. An empty overlap, or a zero cosine norm, is
    ``+inf``.

    Args:
        query: Semantic row of length ``n_rows``.
        matrix: Pack of shape ``(n_individuals, n_rows)``, or one row of
            length ``n_rows`` for a scalar distance.
        metric: ``euclidean`` or ``cosine`` (``1 -`` cosine similarity).
        valid: Optional per-row warmup mask of length ``n_rows``.

    Returns:
        Distances of length ``n_individuals``, or one float when
        ``matrix`` is a single row.

    Raises:
        ValueError: If shapes do not match or ``metric`` is unknown.
    """
    packed = numpy.asarray(matrix, dtype=numpy.float64)
    scalar = packed.ndim == 1
    if scalar:
        packed = packed.reshape(1, -1)
    packed = as_semantic_matrix(packed)
    row = _query_row(query, packed.shape[1])
    dist = _masked_distance(packed, row, _pair_mask(packed, row, valid), metric)
    if scalar:
        return float(dist[0])
    return dist


def semantic_nearest(
    query: numpy.ndarray | Sequence[float],
    matrix: numpy.ndarray | Sequence[Sequence[float]],
    *,
    k: int = 1,
    metric: SemanticMetric = "euclidean",
    valid: numpy.ndarray | None = None,
    individuals: Sequence[Any] | None = None,
    trust_matrix: bool = False,
) -> numpy.ndarray:
    """Return the lowest-index nearest neighbors of ``query``.

    Infinite distances are skipped. Ties keep the lowest pack index.

    Args:
        query: Semantic row of length ``n_rows``.
        matrix: Pack of shape ``(n_individuals, n_rows)``.
        k: Maximum number of neighbors to return.
        metric: ``euclidean`` or ``cosine``.
        valid: Optional per-row warmup mask of length ``n_rows``.
        individuals: Optional population used by ``trust_matrix``.
        trust_matrix: When ``True``, accept ``matrix`` on shape alone.

    Returns:
        Neighbor indices in increasing distance order, length at most
        ``k``. Empty when every distance is infinite.

    Raises:
        ValueError: If ``k`` is less than 1, or the pack does not match
            ``individuals``.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if individuals is None:
        packed = as_semantic_matrix(matrix)
    else:
        packed = validate_semantic_matrix(matrix, individuals, trust_matrix=trust_matrix)
    dist = semantic_distance(query, packed, metric=metric, valid=valid)
    finite = numpy.flatnonzero(numpy.isfinite(dist))
    if finite.size == 0:
        return numpy.empty(0, dtype=int)
    order = finite[numpy.argsort(dist[finite], kind="stable")]
    return order[:k]

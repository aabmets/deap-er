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

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy

if TYPE_CHECKING:
    from deap_er.private.records.archive_common import MapElitesArchive
    from deap_er.private.typedefs import Individual
    from deap_er.private.various.semantic_neighbors import SemanticMetric

from deap_er.private.various.semantic_neighbors import semantic_distance

from .sel_various import sel_random

__all__: list[str] = ["sel_novelty"]


def _archive_descriptor_matrix(
    archive: MapElitesArchive,
    descriptor_fn: Callable[[Any], Sequence[float]],
) -> numpy.ndarray:
    stored = getattr(archive, "descriptors", None)
    if stored is not None:
        matrix = numpy.asarray(stored, dtype=numpy.float64)
        if matrix.size:
            return matrix
    count = len(archive)
    if count == 0:
        return numpy.empty((0, 0), dtype=numpy.float64)
    elites = archive.random_elites(count, replace=False)
    rows = [numpy.asarray(descriptor_fn(elite), dtype=numpy.float64) for elite in elites]
    return numpy.vstack(rows)


def _novelty_score(
    query: numpy.ndarray,
    archive_matrix: numpy.ndarray,
    *,
    k: int,
    metric: SemanticMetric,
    valid: numpy.ndarray | None,
) -> float:
    distances = semantic_distance(query, archive_matrix, metric=metric, valid=valid)
    finite = distances[numpy.isfinite(distances)]
    if finite.size == 0:
        return float("inf")
    take = min(k, finite.size)
    nearest = numpy.partition(finite, take - 1)[:take]
    return float(numpy.mean(nearest))


def sel_novelty(
    individuals: list[Individual],
    sel_count: int,
    archive: MapElitesArchive,
    descriptor_fn: Callable[[Individual], Sequence[float]],
    *,
    k: int = 15,
    metric: SemanticMetric = "euclidean",
    valid: numpy.ndarray | None = None,
) -> list[Individual]:
    """Select individuals with the highest average distance to archive elites.

    Novelty is the mean distance to the ``k`` nearest stored behavior
    descriptors. ``ind.fitness`` is not rewritten; only the ranking key
    changes. An empty archive falls back to uniform random selection.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to return. Non-positive values
            return an empty list.
        archive: MAP-Elites archive whose behavior coordinates define
            the reference set. Uses ``descriptors`` when present, else
            ``descriptor_fn`` on each stored elite.
        descriptor_fn: Maps a pool member to its behavior coordinates.
        k: Number of nearest archive neighbors averaged into the score.
        metric: ``euclidean`` or ``cosine`` (same contract as
            ``semantic_distance``).
        valid: Optional per-dimension warmup mask passed to
            ``semantic_distance``.

    Returns:
        The most novel individuals. Ties keep the lowest pool index.

    Raises:
        ValueError: If ``k`` is less than 1.
    """
    if sel_count <= 0:
        return []
    if not individuals:
        return []
    if k < 1:
        raise ValueError("k must be at least 1")

    archive_matrix = _archive_descriptor_matrix(archive, descriptor_fn)
    if archive_matrix.size == 0:
        return sel_random(individuals, sel_count)

    scores: list[tuple[float, int]] = []
    for index, individual in enumerate(individuals):
        query = numpy.asarray(descriptor_fn(individual), dtype=numpy.float64)
        novelty = _novelty_score(
            query,
            archive_matrix,
            k=k,
            metric=metric,
            valid=valid,
        )
        scores.append((novelty, index))

    order = sorted(scores, key=lambda item: (-item[0], item[1]))
    return [individuals[index] for _, index in order[:sel_count]]

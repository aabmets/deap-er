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

import numpy
from scipy.cluster.vq import kmeans

from deap_er.private.various.rng import rng

__all__: list[str] = ["cvt_centroids", "parse_centroids"]


def parse_centroids(centroids: Sequence[Sequence[float]] | numpy.ndarray) -> numpy.ndarray:
    """Validate and copy a ``(k, dims)`` centroid array.

    Args:
        centroids: Candidate cell centers in descriptor space.

    Returns:
        A copy of the centroids as a contiguous finite unique array.

    Raises:
        ValueError: If the array is empty, not 2-D, non-finite, or
            contains duplicate rows.
    """
    array = numpy.array(centroids, dtype=numpy.float64, copy=True, order="C")
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("centroids must be a non-empty 2-D array")
    if not bool(numpy.isfinite(array).all()):
        raise ValueError("centroids must be finite")
    if numpy.unique(array, axis=0).shape[0] != array.shape[0]:
        raise ValueError("centroids must be unique")
    return array


def cvt_centroids(
    samples: Sequence[Sequence[float]] | numpy.ndarray,
    k: int,
    *,
    n_iter: int = 20,
) -> numpy.ndarray:
    """Compute ``k`` centroids by k-means on a behavior sample.

    Seeding uses the process-wide ``tools.rng`` generator so
    checkpointed runs stay reproducible. Callers who already have
    centroids can pass them straight to
    :class:`~deap_er.records.CvtArchive`.

    Args:
        samples: Behavior descriptors with shape ``(n, dims)``.
        k: Number of centroids. Must be at least 1 and at most ``n``.
        n_iter: Independent k-means runs; the lowest-distortion result
            is kept.

    Returns:
        Contiguous ``(k, dims)`` array of centroids.

    Raises:
        ValueError: If ``samples`` is empty, not 2-D, or non-finite,
            or if ``k`` or ``n_iter`` is invalid.
    """
    array = numpy.asarray(samples, dtype=numpy.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("samples must be a non-empty 2-D array")
    if not bool(numpy.isfinite(array).all()):
        raise ValueError("samples must be finite")
    k_value = int(k)
    if k_value < 1:
        raise ValueError("k must be at least 1")
    if k_value > array.shape[0]:
        raise ValueError("k cannot exceed the number of samples")
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")
    seed = int(rng.integers(0, 2**31))
    centroids, _distortion = kmeans(array, k_value, iter=n_iter, rng=seed)
    result = numpy.ascontiguousarray(centroids, dtype=numpy.float64)
    if result.shape != (k_value, array.shape[1]):
        raise ValueError(f"k-means returned {result.shape[0]} centroids, expected {k_value}")
    return result

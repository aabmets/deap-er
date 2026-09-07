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
from typing import Any

import numpy

from deap_er.private.various.rng import rng
from deap_er.private.various.semantic_mask import (
    as_semantic_matrix,
    packed_semantics,
    semantic_column_keep,
)

__all__: list[str] = ["semantic_pca_basis", "semantic_project", "semantic_random_basis"]


def semantic_random_basis(n_rows: int, n_dims: int) -> numpy.ndarray:
    """Return a Gaussian random-projection basis.

    Columns are scaled by ``1 / sqrt(n_dims)``. Draws use the process
    :data:`~deap_er.tools.rng`.

    Args:
        n_rows: Number of semantic coordinates (rows of the pack).
        n_dims: Number of projected dimensions.

    Returns:
        Basis of shape ``(n_rows, n_dims)``.

    Raises:
        ValueError: If ``n_rows`` or ``n_dims`` is not positive.
    """
    if n_rows < 1 or n_dims < 1:
        raise ValueError("n_rows and n_dims must be positive")
    draw = rng.standard_normal((n_rows, n_dims))
    return draw / numpy.sqrt(n_dims)


def semantic_pca_basis(
    matrix: numpy.ndarray | Sequence[Sequence[float]],
    n_dims: int,
    *,
    valid: numpy.ndarray | None = None,
    target: numpy.ndarray | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return a thin-SVD basis and column center for :func:`semantic_project`.

    Columns outside ``valid`` (and non-finite ``target`` samples) are
    dropped before centering. Unused rows of the returned basis and
    center are zero.

    Args:
        matrix: Semantic pack of shape ``(n_individuals, n_rows)``.
        n_dims: Number of components to keep.
        valid: Optional per-row warmup mask of length ``n_rows``.
        target: Optional target whose non-finite samples are dropped.

    Returns:
        ``(basis, center)`` with shapes ``(n_rows, n_dims)`` and
        ``(n_rows,)``.

    Raises:
        ValueError: If ``n_dims`` is not positive, or no finite row
            remains on the kept columns.
    """
    if n_dims < 1:
        raise ValueError("n_dims must be positive")
    packed = as_semantic_matrix(matrix)
    keep = semantic_column_keep(packed.shape[1], valid, target)
    kept = packed[:, keep]
    finite_rows = numpy.all(numpy.isfinite(kept), axis=1) if kept.size else numpy.zeros(0, dtype=bool)
    kept = kept[finite_rows]
    if kept.shape[0] == 0 or kept.shape[1] == 0:
        raise ValueError("semantic_pca_basis needs at least one finite row on kept columns")
    center_keep = kept.mean(axis=0)
    _, _, vt = numpy.linalg.svd(kept - center_keep, full_matrices=False)
    n_comp = min(n_dims, vt.shape[0])
    basis = numpy.zeros((packed.shape[1], n_dims), dtype=numpy.float64)
    center = numpy.zeros(packed.shape[1], dtype=numpy.float64)
    center[keep] = center_keep
    if n_comp:
        basis[numpy.ix_(numpy.flatnonzero(keep), numpy.arange(n_comp))] = vt[:n_comp].T
    return basis, center


def semantic_project(
    matrix: numpy.ndarray | Sequence[Sequence[float]],
    basis: numpy.ndarray,
    *,
    valid: numpy.ndarray | None = None,
    target: numpy.ndarray | None = None,
    center: numpy.ndarray | None = None,
    individuals: Sequence[Any] | None = None,
    trust_matrix: bool = False,
) -> numpy.ndarray:
    """Project a semantic pack through a caller-supplied basis.

    Columns outside ``valid`` (and non-finite ``target`` samples) are
    zeroed so warmup does not enter the product. A row that is still
    non-finite on a kept column yields a non-finite descriptor.

    Args:
        matrix: Semantic pack of shape ``(n_individuals, n_rows)``.
        basis: Projection matrix of shape ``(n_rows, n_dims)``.
        valid: Optional per-row warmup mask of length ``n_rows``.
        target: Optional target whose non-finite samples are dropped.
        center: Optional length-``n_rows`` vector subtracted before the
            product. Used with :func:`semantic_pca_basis`.
        individuals: Optional population used by ``trust_matrix``.
        trust_matrix: When ``True``, accept ``matrix`` on shape alone.

    Returns:
        Descriptor array of shape ``(n_individuals, n_dims)``.

    Raises:
        ValueError: If ``basis`` or ``center`` does not match ``n_rows``.
    """
    packed = packed_semantics(matrix, individuals, trust_matrix)
    components = numpy.asarray(basis, dtype=numpy.float64)
    if components.ndim != 2 or components.shape[0] != packed.shape[1]:
        raise ValueError(
            f"basis must have shape ({packed.shape[1]}, n_dims), got {components.shape}"
        )
    keep = semantic_column_keep(packed.shape[1], valid, target)
    shifted = packed.copy()
    if center is not None:
        mean = numpy.asarray(center, dtype=numpy.float64)
        if mean.ndim != 1 or mean.shape[0] != packed.shape[1]:
            raise ValueError("center must be a one-dimensional vector matching n_rows")
        shifted = shifted - mean
    shifted[:, ~keep] = 0.0
    return shifted @ components

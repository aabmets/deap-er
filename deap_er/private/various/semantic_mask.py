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

__all__: list[str] = [
    "as_semantic_matrix",
    "packed_semantics",
    "semantic_column_keep",
    "semantic_valid_mask",
    "validate_semantic_matrix",
]


def as_semantic_matrix(matrix: numpy.ndarray | Sequence[Sequence[float]]) -> numpy.ndarray:
    """Pack a semantic matrix as C-contiguous ``float64``.

    Args:
        matrix: Caller-supplied ``(n_individuals, n_rows)`` pack.

    Returns:
        A two-dimensional ``float64`` array.

    Raises:
        ValueError: If ``matrix`` is not two-dimensional.
    """
    packed = numpy.asarray(matrix, dtype=numpy.float64)
    if packed.ndim != 2:
        raise ValueError(
            "semantic matrix must be two-dimensional "
            f"(n_individuals, n_rows), got ndim={packed.ndim}"
        )
    return packed


def semantic_valid_mask(
    matrix: numpy.ndarray | Sequence[Sequence[float]],
    valid: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """Return the finite sample mask for a semantic matrix.

    A one-dimensional ``valid`` is broadcast across individuals and
    intersected with ``isfinite(matrix)``. A sample that is ``True`` in
    ``valid`` but non-finite still stays out of the mask.

    Args:
        matrix: Semantic pack of shape ``(n_individuals, n_rows)``.
        valid: Optional per-row warmup mask of length ``n_rows``.

    Returns:
        Boolean mask with the same shape as ``matrix``.

    Raises:
        ValueError: If ``matrix`` is not two-dimensional, or ``valid`` is
            not a one-dimensional mask of length ``n_rows``.
    """
    packed = as_semantic_matrix(matrix)
    finite = numpy.isfinite(packed)
    if valid is None:
        return finite
    sample_valid = numpy.asarray(valid, dtype=bool)
    if sample_valid.ndim != 1 or sample_valid.shape[0] != packed.shape[1]:
        raise ValueError("valid must be a one-dimensional mask matching the series length")
    return sample_valid[numpy.newaxis, :] & finite


def validate_semantic_matrix(
    matrix: numpy.ndarray | Sequence[Sequence[float]],
    individuals: Sequence[Any] | None,
    *,
    trust_matrix: bool = False,
) -> numpy.ndarray:
    """Check that a semantic pack is row-aligned with ``individuals``.

    Unlike lexicase, the series cannot be compared to ``fitness.values``.
    When ``trust_matrix`` is ``False``, every individual must have a
    valid fitness. When ``True``, only the leading shape is checked.

    Args:
        matrix: Semantic pack of shape ``(n_individuals, n_rows)``.
        individuals: Population the rows describe, or ``None`` to skip
            the alignment check.
        trust_matrix: When ``True``, accept the pack on shape alone.

    Returns:
        The packed ``float64`` matrix.

    Raises:
        ValueError: If ``matrix`` is not two-dimensional, its leading
            length does not match ``individuals``, or a fitness is
            missing when ``trust_matrix`` is ``False``.
    """
    packed = as_semantic_matrix(matrix)
    if individuals is None:
        return packed
    expected = (len(individuals), packed.shape[1])
    if packed.shape[0] != expected[0]:
        raise ValueError(f"matrix must have shape {expected}, got {packed.shape}")
    if trust_matrix:
        return packed
    for individual in individuals:
        fitness = getattr(individual, "fitness", None)
        if fitness is None or not fitness.is_valid():
            raise ValueError("every individual must have a valid fitness")
    return packed


def packed_semantics(
    matrix: numpy.ndarray | Sequence[Sequence[float]],
    individuals: Sequence[Any] | None,
    trust_matrix: bool,
) -> numpy.ndarray:
    """Pack a semantic matrix, optionally checking ``individuals``.

    Args:
        matrix: Semantic pack of shape ``(n_individuals, n_rows)``.
        individuals: Population the rows describe, or ``None``.
        trust_matrix: When ``True``, accept the pack on shape alone.

    Returns:
        The packed ``float64`` matrix.
    """
    if individuals is None:
        return as_semantic_matrix(matrix)
    return validate_semantic_matrix(matrix, individuals, trust_matrix=trust_matrix)


def semantic_column_keep(
    n_rows: int,
    valid: numpy.ndarray | None,
    target: numpy.ndarray | None,
) -> numpy.ndarray:
    """Return the shared column mask used by projection helpers.

    Args:
        n_rows: Number of semantic coordinates.
        valid: Optional per-row warmup mask of length ``n_rows``.
        target: Optional target whose non-finite samples are dropped.

    Returns:
        One-dimensional ``bool`` mask of kept columns.

    Raises:
        ValueError: If ``valid`` or ``target`` does not match ``n_rows``.
    """
    keep = numpy.ones(n_rows, dtype=bool) if valid is None else numpy.asarray(valid, dtype=bool)
    if keep.ndim != 1 or keep.shape[0] != n_rows:
        raise ValueError("valid must be a one-dimensional mask matching the series length")
    if target is None:
        return keep
    series = numpy.asarray(target, dtype=numpy.float64)
    if series.ndim != 1 or series.shape[0] != n_rows:
        raise ValueError("target must be a one-dimensional series matching n_rows")
    return keep & numpy.isfinite(series)

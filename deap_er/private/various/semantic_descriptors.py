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

from deap_er.private.various.case_errors import case_intervals
from deap_er.private.various.semantic_mask import packed_semantics, semantic_valid_mask
from deap_er.private.various.semantic_project import semantic_project

__all__: list[str] = ["semantic_descriptors", "semantic_moments", "semantic_solve_bits"]

type DescriptorKind = Literal["moments", "solve", "project"]


def _row_moments(matrix: numpy.ndarray, row_valid: numpy.ndarray) -> numpy.ndarray:
    out = numpy.full((matrix.shape[0], 4), numpy.nan, dtype=numpy.float64)
    ok = row_valid.any(axis=1)
    if not numpy.any(ok):
        return out
    filled = numpy.where(row_valid, matrix, numpy.nan)
    with numpy.errstate(invalid="ignore", divide="ignore"):
        out[ok, 0] = numpy.nanmean(filled[ok], axis=1)
        out[ok, 1] = numpy.nanstd(filled[ok], axis=1, ddof=0)
        out[ok, 2] = numpy.nanmin(filled[ok], axis=1)
        out[ok, 3] = numpy.nanmax(filled[ok], axis=1)
    return out


def semantic_moments(
    matrix: numpy.ndarray | Sequence[Sequence[float]],
    *,
    valid: numpy.ndarray | None = None,
    individuals: Sequence[Any] | None = None,
    trust_matrix: bool = False,
) -> numpy.ndarray:
    """Return per-individual mean, population std, min, and max.

    Moments use samples where ``valid`` (broadcast) and the row are
    finite. An empty row is all-NaN. A single sample has ``std = 0``.

    Args:
        matrix: Semantic pack of shape ``(n_individuals, n_rows)``.
        valid: Optional per-row warmup mask of length ``n_rows``.
        individuals: Optional population used by ``trust_matrix``.
        trust_matrix: When ``True``, accept ``matrix`` on shape alone.

    Returns:
        Descriptor array of shape ``(n_individuals, 4)``.
    """
    packed = packed_semantics(matrix, individuals, trust_matrix)
    return _row_moments(packed, semantic_valid_mask(packed, valid))


def semantic_solve_bits(
    matrix: numpy.ndarray | Sequence[Sequence[float]],
    target: numpy.ndarray,
    ranges: Sequence[tuple[int, int]] | numpy.ndarray,
    *,
    valid: numpy.ndarray | None = None,
    empty: float = float("inf"),
    individuals: Sequence[Any] | None = None,
    trust_matrix: bool = False,
) -> numpy.ndarray:
    """Return one solved-case bit per individual and case.

    A case is solved when its MSE is ``isclose`` to ``0`` with
    ``abs_tol=1e-12``, matching :func:`~deap_er.tools.sample_informed_cases`.
    Segment bounds and ``valid=`` follow :func:`~deap_er.tools.case_errors`.

    Args:
        matrix: Predicted pack of shape ``(n_individuals, n_rows)``.
        target: Target series of length ``n_rows``.
        ranges: Case bounds accepted by :func:`~deap_er.tools.case_intervals`.
        valid: Optional per-row warmup mask of length ``n_rows``.
        empty: MSE used when a case has no scorable samples.
        individuals: Optional population used by ``trust_matrix``.
        trust_matrix: When ``True``, accept ``matrix`` on shape alone.

    Returns:
        Float ``0/1`` array of shape ``(n_individuals, n_cases)``.
    """
    packed = packed_semantics(matrix, individuals, trust_matrix)
    series = numpy.asarray(target, dtype=numpy.float64)
    if series.ndim != 1 or series.shape[0] != packed.shape[1]:
        raise ValueError("target must be a one-dimensional series matching n_rows")
    intervals = case_intervals(ranges, packed.shape[1])
    row_valid = semantic_valid_mask(packed, valid) & numpy.isfinite(series)
    bits = numpy.empty((packed.shape[0], len(intervals)), dtype=numpy.float64)
    for index, (start, stop) in enumerate(intervals):
        sample = row_valid[:, start:stop]
        diff = packed[:, start:stop] - series[start:stop]
        sq = numpy.where(sample, diff * diff, 0.0)
        counts = sample.sum(axis=1)
        mse = numpy.where(counts > 0, sq.sum(axis=1) / counts, empty)
        bits[:, index] = numpy.isclose(mse, 0.0, atol=1e-12)
    return bits


def semantic_descriptors(
    matrix: numpy.ndarray | Sequence[Sequence[float]],
    *,
    kind: DescriptorKind = "moments",
    valid: numpy.ndarray | None = None,
    target: numpy.ndarray | None = None,
    ranges: Sequence[tuple[int, int]] | numpy.ndarray | None = None,
    basis: numpy.ndarray | None = None,
    center: numpy.ndarray | None = None,
    empty: float = float("inf"),
    individuals: Sequence[Any] | None = None,
    trust_matrix: bool = False,
) -> numpy.ndarray:
    """Dispatch a semantic pack to moments, solve bits, or a projection.

    Args:
        matrix: Semantic pack of shape ``(n_individuals, n_rows)``.
        kind: ``moments``, ``solve``, or ``project``.
        valid: Optional per-row warmup mask of length ``n_rows``.
        target: Target series. Required for ``solve``.
        ranges: Case bounds. Required for ``solve``.
        basis: Projection matrix. Required for ``project``.
        center: Optional center passed to :func:`semantic_project`.
        empty: Empty-case MSE for ``solve``.
        individuals: Optional population used by ``trust_matrix``.
        trust_matrix: When ``True``, accept ``matrix`` on shape alone.

    Returns:
        Descriptor array whose width depends on ``kind``.

    Raises:
        ValueError: If ``kind`` is unknown or a required argument is
            missing.
    """
    if kind == "moments":
        return semantic_moments(
            matrix, valid=valid, individuals=individuals, trust_matrix=trust_matrix
        )
    if kind == "solve":
        if target is None or ranges is None:
            raise ValueError("kind='solve' requires target and ranges")
        return semantic_solve_bits(
            matrix,
            target,
            ranges,
            valid=valid,
            empty=empty,
            individuals=individuals,
            trust_matrix=trust_matrix,
        )
    if kind == "project":
        if basis is None:
            raise ValueError("kind='project' requires basis")
        return semantic_project(
            matrix,
            basis,
            valid=valid,
            target=target,
            center=center,
            individuals=individuals,
            trust_matrix=trust_matrix,
        )
    raise ValueError(f"unknown descriptor kind {kind!r}")

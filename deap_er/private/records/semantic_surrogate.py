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

from deap_er.private.various.semantic_mask import as_semantic_matrix, validate_semantic_matrix
from deap_er.private.various.semantic_neighbors import semantic_nearest

__all__: list[str] = ["SemanticSurrogate"]

type SurrogateKind = Literal["nearest", "linear"]
type SemanticMetric = Literal["euclidean", "cosine"]


class SemanticSurrogate:
    """Last-generation semantic store for nearest and linear lookup.

    ``update`` replaces the stored pack and scalar targets. It does not
    write ``ind.fitness``. ``predict`` is a stand-in for last-generation
    semantics, not a learned quality-diversity model.

    Args:
        metric: Default finite-mask distance for nearest lookup.
    """

    def __init__(self, *, metric: SemanticMetric = "euclidean") -> None:
        """See the class docstring."""
        self._metric: SemanticMetric = metric
        self._matrix: numpy.ndarray | None = None
        self._values: numpy.ndarray | None = None
        self._valid: numpy.ndarray | None = None

    @property
    def metric(self) -> SemanticMetric:
        """Default nearest-neighbor metric."""
        return self._metric

    def update(
        self,
        matrix: numpy.ndarray | Sequence[Sequence[float]],
        values: numpy.ndarray | Sequence[float],
        *,
        valid: numpy.ndarray | None = None,
        individuals: Sequence[Any] | None = None,
        trust_matrix: bool = False,
    ) -> None:
        """Replace the stored last-generation pack.

        Args:
            matrix: Semantic pack of shape ``(n_individuals, n_rows)``.
            values: Scalar target per row, typically ``fitness.wvalues[0]``.
            valid: Optional per-row warmup mask stored with the pack.
            individuals: Optional population used by ``trust_matrix``.
            trust_matrix: When ``True``, accept ``matrix`` on shape alone.

        Raises:
            ValueError: If ``values`` length does not match the pack, or
                the pack does not match ``individuals``.
        """
        if individuals is None:
            packed = as_semantic_matrix(matrix)
        else:
            packed = validate_semantic_matrix(matrix, individuals, trust_matrix=trust_matrix)
        stored = numpy.asarray(values, dtype=numpy.float64)
        if stored.ndim != 1 or stored.shape[0] != packed.shape[0]:
            raise ValueError("values must be a one-dimensional vector matching n_individuals")
        if valid is not None:
            sample_valid = numpy.asarray(valid, dtype=bool)
            if sample_valid.ndim != 1 or sample_valid.shape[0] != packed.shape[1]:
                raise ValueError("valid must be a one-dimensional mask matching the series length")
            self._valid = sample_valid.copy()
        else:
            self._valid = None
        self._matrix = packed.copy()
        self._values = stored.copy()

    def nearest(
        self,
        query: numpy.ndarray | Sequence[float],
        k: int = 1,
        *,
        metric: SemanticMetric | None = None,
        valid: numpy.ndarray | None = None,
    ) -> numpy.ndarray:
        """Return stored-row indices nearest to ``query``.

        Args:
            query: Semantic row of length ``n_rows``.
            k: Maximum number of neighbors to return.
            metric: Distance used for this call. Defaults to the
                constructor metric.
            valid: Warmup mask. Defaults to the mask from ``update``.

        Returns:
            Neighbor indices in increasing distance order.

        Raises:
            ValueError: If the store is empty.
        """
        packed, _values = self._require_store()
        mask = self._valid if valid is None else valid
        return semantic_nearest(
            query,
            packed,
            k=k,
            metric=self._metric if metric is None else metric,
            valid=mask,
        )

    def predict(
        self,
        query: numpy.ndarray | Sequence[float],
        *,
        kind: SurrogateKind = "nearest",
        k: int = 1,
        metric: SemanticMetric | None = None,
        valid: numpy.ndarray | None = None,
    ) -> float:
        """Predict a scalar from last-generation semantics.

        ``nearest`` returns the stored value of the nearest row, or the
        mean of ``k`` neighbors. ``linear`` fits least squares on finite
        stored rows. Fallback to nearest happens only when the design is
        empty or ``rank < 1``, not when ``rank < min(shape)``.
        Underdetermined packs (more columns than rows) keep the
        minimum-norm solution.

        Args:
            query: Semantic row of length ``n_rows``.
            kind: ``nearest`` or ``linear``.
            k: Neighbor count for ``nearest`` (and the linear fallback).
            metric: Distance used for nearest lookup.
            valid: Warmup mask. Defaults to the mask from ``update``.

        Returns:
            Predicted scalar, or ``nan`` when no finite neighbor exists.

        Raises:
            ValueError: If the store is empty or ``kind`` is unknown.
        """
        if kind == "nearest":
            return self._predict_nearest(query, k=k, metric=metric, valid=valid)
        if kind == "linear":
            return self._predict_linear(query, k=k, metric=metric, valid=valid)
        raise ValueError(f"kind must be 'nearest' or 'linear', got {kind!r}")

    def _require_store(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        if self._matrix is None or self._values is None:
            raise ValueError("SemanticSurrogate.update must be called before lookup")
        return self._matrix, self._values

    def _predict_nearest(
        self,
        query: numpy.ndarray | Sequence[float],
        *,
        k: int,
        metric: SemanticMetric | None,
        valid: numpy.ndarray | None,
    ) -> float:
        _packed, values = self._require_store()
        neighbors = self.nearest(query, k=k, metric=metric, valid=valid)
        if neighbors.size == 0:
            return float("nan")
        return float(values[neighbors].mean())

    def _predict_linear(
        self,
        query: numpy.ndarray | Sequence[float],
        *,
        k: int,
        metric: SemanticMetric | None,
        valid: numpy.ndarray | None,
    ) -> float:
        packed, values = self._require_store()
        mask = self._valid if valid is None else valid
        row = numpy.asarray(query, dtype=numpy.float64)
        if row.ndim != 1 or row.shape[0] != packed.shape[1]:
            raise ValueError(f"query must have shape ({packed.shape[1]},), got {row.shape}")
        if mask is None:
            keep = numpy.ones(packed.shape[1], dtype=bool)
        else:
            keep = numpy.asarray(mask, dtype=bool)
        if keep.ndim != 1 or keep.shape[0] != packed.shape[1]:
            raise ValueError("valid must be a one-dimensional mask matching the series length")
        keep = keep & numpy.isfinite(row)
        design = packed[:, keep]
        usable = numpy.all(numpy.isfinite(design), axis=1) & numpy.isfinite(values)
        design = design[usable]
        target = values[usable]
        probe = row[keep]
        if design.shape[0] == 0 or design.shape[1] == 0:
            return self._predict_nearest(query, k=k, metric=metric, valid=valid)
        solution, _residuals, rank, _singular = numpy.linalg.lstsq(design, target, rcond=None)
        if rank < 1:
            return self._predict_nearest(query, k=k, metric=metric, valid=valid)
        return float(probe @ solution)

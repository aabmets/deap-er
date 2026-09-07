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

from collections.abc import Sequence
from numbers import Integral

import numpy

from deap_er.private.various.case_bounds import (
    mask_from_ranges,
    normalize_case_ranges,
    ranges_from_mask,
)

__all__: list[str] = ["CaseExam", "CaseRanges"]

type CaseRanges = Sequence[tuple[int, int]] | numpy.ndarray


class CaseExam:
    """A case subset stored as ranges or a 1-D bool mask.

    This is data, not a genome. The same shapes feed ``case_errors``
    (series segments) and ``sel_lexicase(..., cases=)`` (catalog indices
    via :meth:`as_cases`).
    """

    def __init__(
        self,
        ranges: CaseRanges | None = None,
        mask: numpy.ndarray | None = None,
    ) -> None:
        """Store exactly one of ``ranges`` or ``mask``.

        Args:
            ranges: Half-open ``(start, stop)`` pairs or a ``(n, 2)``
                integer table.
            mask: One-dimensional ``bool`` array.

        Raises:
            ValueError: If both or neither form is given, or ``mask`` is
                not a 1-D ``bool`` array.
        """
        if (ranges is None) == (mask is None):
            raise ValueError("CaseExam requires exactly one of ranges or mask")
        self._ranges: list[tuple[int, int]] | None
        self._mask: numpy.ndarray | None
        if mask is not None:
            array = numpy.asarray(mask)
            if array.ndim != 1:
                raise ValueError("a boolean mask must be one-dimensional")
            if array.dtype != bool:
                raise ValueError("a boolean mask must have dtype bool")
            self._mask = array.copy()
            self._ranges = None
            return
        self._mask = None
        self._ranges = [tuple(pair) for pair in ranges]  # type: ignore[union-attr]

    @classmethod
    def from_cases(cls, cases: Sequence[int], n_cases: int) -> CaseExam:
        """Build a catalog-index exam from selected case indices.

        Args:
            cases: Fitness-case indices to mark ``True``.
            n_cases: Length of the catalog mask.

        Returns:
            An exam whose mask has length ``n_cases``.

        Raises:
            IndexError: If an index is not a valid case index.
            ValueError: If ``n_cases`` is negative.
        """
        if n_cases < 0:
            raise ValueError("n_cases must be non-negative")
        mask = numpy.zeros(n_cases, dtype=bool)
        for idx in cases:
            mask[_case_index(idx, n_cases)] = True
        return cls(mask=mask)

    @property
    def ranges(self) -> list[tuple[int, int]] | None:
        """Live range list, or ``None`` when the exam stores a mask."""
        return self._ranges

    @property
    def mask(self) -> numpy.ndarray | None:
        """Live boolean mask, or ``None`` when the exam stores ranges."""
        return self._mask

    def as_ranges(self, length: int) -> list[tuple[int, int]]:
        """Return validated ``[start, stop)`` intervals against ``length``.

        Args:
            length: Exclusive upper bound for endpoints, or the mask length.

        Returns:
            Half-open intervals in stored order.

        Raises:
            ValueError: If stored bounds or the mask do not match ``length``.
        """
        if self._mask is not None:
            return ranges_from_mask(self._mask, length)
        return normalize_case_ranges(self._ranges or [], length)

    def as_mask(self, length: int) -> numpy.ndarray:
        """Return a boolean mask of ``length``.

        Args:
            length: Length of the painted mask, or the stored mask length.

        Returns:
            A 1-D ``bool`` copy (stored masks) or a painted range mask.

        Raises:
            ValueError: If stored bounds or the mask do not match ``length``.
        """
        if self._mask is not None:
            if self._mask.shape[0] != length:
                raise ValueError("a boolean mask must match the series length")
            return self._mask.copy()
        return mask_from_ranges(self._ranges or [], length)

    def as_cases(self, n_cases: int) -> list[int]:
        """Interpret the exam as a subset of ``n_cases`` catalog indices.

        A stored mask must have length ``n_cases``. Stored ranges are
        treated as half-open intervals over those indices.

        Args:
            n_cases: Number of fitness cases in the current pack.

        Returns:
            Distinct case indices in first-occurrence order.

        Raises:
            ValueError: If stored bounds or the mask do not match ``n_cases``.
        """
        if self._mask is not None:
            if self._mask.shape[0] != n_cases:
                raise ValueError("a boolean mask must match the series length")
            return [int(idx) for idx in numpy.flatnonzero(self._mask)]
        seen: set[int] = set()
        chosen: list[int] = []
        for start, stop in normalize_case_ranges(self._ranges or [], n_cases):
            for idx in range(start, stop):
                if idx not in seen:
                    seen.add(idx)
                    chosen.append(idx)
        return chosen

    def copy(self) -> CaseExam:
        """Return a copy of the stored ranges or mask.

        Returns:
            A new exam with the same subset.
        """
        if self._mask is not None:
            return CaseExam(mask=self._mask)
        return CaseExam(ranges=list(self._ranges or []))

    def assign(self, other: CaseExam) -> None:
        """Replace this exam's storage with a copy of ``other``.

        Args:
            other: Exam whose ranges or mask become this exam's data.
        """
        if other._mask is not None:
            self._mask = other._mask.copy()
            self._ranges = None
            return
        self._mask = None
        self._ranges = list(other._ranges or [])


def _case_index(idx: object, n_cases: int) -> int:
    if isinstance(idx, bool) or not isinstance(idx, Integral):
        raise IndexError(f"case index {idx} is out of range for {n_cases} fitness cases")
    value = int(idx)
    if value < 0 or value >= n_cases:
        raise IndexError(f"case index {value} is out of range for {n_cases} fitness cases")
    return value

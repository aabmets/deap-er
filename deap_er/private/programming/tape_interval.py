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
from dataclasses import dataclass

import numpy

from .tape import Tape
from .tape_interval_ops import normalize_bounds
from .tape_interval_walk import walk_tape

__all__: list[str] = [
    "TapeFlags",
    "bounds_from_matrix",
    "tape_flags",
    "tape_interval",
    "tape_skip_score",
]


@dataclass(frozen=True)
class TapeFlags:
    """Static certificates for a lowered tape."""

    all_nan: bool
    constant: bool
    hides_warmup: bool

    @property
    def skip_score(self) -> bool:
        """Return whether scoring should skip ``interpret_tapes``."""
        return self.all_nan or self.constant or self.hides_warmup


def bounds_from_matrix(matrix: numpy.ndarray | Sequence[Sequence[float]]) -> numpy.ndarray:
    """Return empirical ``(low, high)`` bounds for each matrix column.

    Args:
        matrix: Packed ``(n_rows, n_columns)`` table.

    Returns:
        A ``(n_columns, 2)`` ``float64`` array of column bounds.

    Raises:
        ValueError: If ``matrix`` is not two-dimensional.
    """
    packed = numpy.asarray(matrix, dtype=numpy.float64)
    if packed.ndim != 2:
        raise ValueError("matrix must be a two-dimensional array")
    bounds = numpy.empty((packed.shape[1], 2), dtype=numpy.float64)
    for column in range(packed.shape[1]):
        series = packed[:, column]
        bounds[column, 0] = numpy.nanmin(series)
        bounds[column, 1] = numpy.nanmax(series)
    return bounds


def tape_interval(
    tape: Tape,
    column_bounds: numpy.ndarray | Sequence[tuple[float, float]],
) -> tuple[float, float]:
    """Return a conservative output interval for a tape.

    Args:
        tape: Tape produced by ``lower_tree``.
        column_bounds: Per-column ``(low, high)`` bounds.

    Returns:
        The ``(low, high)`` envelope of finite outputs.

    Raises:
        ValueError: If the tape is malformed or holds a consumer opcode.
    """
    walked = walk_tape(tape, normalize_bounds(column_bounds, tape.columns))
    return walked.summary.lo, walked.summary.hi


def tape_flags(
    tape: Tape,
    column_bounds: numpy.ndarray | Sequence[tuple[float, float]],
    *,
    n_rows: int,
) -> TapeFlags:
    """Return static certificates for a tape before scoring.

    Args:
        tape: Tape produced by ``lower_tree``.
        column_bounds: Per-column ``(low, high)`` bounds.
        n_rows: Row count of the matrix the tape would run on.

    Returns:
        Flags for identically ``nan``, constant, or warmup-hiding programs.

    Raises:
        ValueError: If ``n_rows`` is negative, the tape is malformed, or
            the tape holds a consumer opcode.
    """
    if n_rows < 0:
        raise ValueError("n_rows must be at least 0.")
    walked = walk_tape(tape, normalize_bounds(column_bounds, tape.columns))
    summary = walked.summary
    scorable = summary.can_finite and summary.first_finite < n_rows
    return TapeFlags(
        all_nan=not scorable,
        constant=scorable and summary.const and summary.lo == summary.hi,
        hides_warmup=walked.warmup_hidden,
    )


def tape_skip_score(
    tape: Tape,
    column_bounds: numpy.ndarray | Sequence[tuple[float, float]],
    *,
    n_rows: int,
) -> bool:
    """Return whether ``evaluate_columnar`` should skip scoring a tape.

    Args:
        tape: Tape produced by ``lower_tree``.
        column_bounds: Per-column ``(low, high)`` bounds.
        n_rows: Row count of the matrix the tape would run on.

    Returns:
        ``True`` when any static certificate fires.

    Raises:
        ValueError: If ``n_rows`` is negative, the tape is malformed, or
            the tape holds a consumer opcode.
    """
    return tape_flags(tape, column_bounds, n_rows=n_rows).skip_score

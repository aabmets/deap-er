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

from .opcode_set import OPCODES_ARITY, USER_BASE, Opcode
from .tape import Tape
from .tape_interval_ops import (
    BINARY,
    BINARY_PROTECTED,
    COMPARISONS,
    PAIR_WINDOWED,
    STACK_UNDERFLOW,
    UNARY,
    WINDOWED,
    Summary,
    apply_binary,
    apply_unary,
    hides_warmup,
    merge_arrays,
    normalize_bounds,
)
from .tape_interval_window import apply_pair_window, apply_window

__all__: list[str] = [
    "TapeFlags",
    "bounds_from_matrix",
    "tape_flags",
    "tape_interval",
    "tape_skip_score",
]

_CONSUMER_INTERVAL = (
    "Consumer opcode {opcode} has no interval certificate. "
    "Rescore the full matrix with interpret_tapes."
)


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


@dataclass(frozen=True)
class _WalkResult:
    summary: Summary
    warmup_hidden: bool


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
    walked = _walk_tape(tape, normalize_bounds(column_bounds, tape.columns))
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
    walked = _walk_tape(tape, normalize_bounds(column_bounds, tape.columns))
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


def _walk_tape(tape: Tape, column_bounds: numpy.ndarray) -> _WalkResult:
    stack: list[Summary] = []
    hides = False
    for step in range(tape.opcodes.size):
        opcode = int(tape.opcodes[step])
        operand = int(tape.operands[step])
        if opcode >= USER_BASE:
            raise ValueError(_CONSUMER_INTERVAL.format(opcode=opcode))
        if opcode == int(Opcode.COL_LOAD):
            lo, hi = float(column_bounds[operand, 0]), float(column_bounds[operand, 1])
            can_finite = numpy.isfinite(lo) or numpy.isfinite(hi)
            stack.append(Summary(lo, hi, 0, 0, lo == hi, can_finite, "array"))
            continue
        if opcode == int(Opcode.CONST):
            value = float(tape.constants[operand])
            stack.append(Summary(value, value, 0, 0, True, numpy.isfinite(value), "array"))
            continue
        if opcode in COMPARISONS:
            right = _pop_array(stack)
            left = _pop_array(stack)
            compared = max(left.lookback, right.lookback)
            stack.append(Summary(0.0, 1.0, 0, 0, False, True, "mask", compared))
            continue
        if opcode == int(Opcode.NOT):
            mask = _pop_mask(stack)
            stack.append(Summary(0.0, 1.0, 0, 0, False, True, "mask", mask.compared_lookback))
            continue
        if opcode in {int(Opcode.AND), int(Opcode.OR)}:
            right = _pop_mask(stack)
            left = _pop_mask(stack)
            stack.append(
                Summary(
                    0.0,
                    1.0,
                    0,
                    0,
                    False,
                    True,
                    "mask",
                    max(left.compared_lookback, right.compared_lookback),
                )
            )
            continue
        if opcode == int(Opcode.WHERE):
            on_false = _pop_array(stack)
            on_true = _pop_array(stack)
            condition = _pop_mask(stack)
            hides |= hides_warmup(condition, on_true, on_false)
            stack.append(merge_arrays(on_true, on_false))
            continue
        if opcode in UNARY:
            child = _pop_array(stack)
            stack.append(apply_unary(opcode, child, tape.fill))
            continue
        if opcode in BINARY or opcode in BINARY_PROTECTED:
            right = _pop_array(stack)
            left = _pop_array(stack)
            stack.append(apply_binary(opcode, left, right, tape.fill))
            continue
        if opcode in WINDOWED:
            child = _pop_array(stack)
            stack.append(apply_window(opcode, child, operand))
            continue
        if opcode in PAIR_WINDOWED:
            right = _pop_array(stack)
            left = _pop_array(stack)
            stack.append(apply_pair_window(opcode, left, right, operand))
            continue
        if opcode not in OPCODES_ARITY:
            raise ValueError(f"Opcode {opcode} has no interval certificate.")
    if not stack:
        raise ValueError("The tape is malformed and leaves no result.")
    root = stack[-1]
    if root.kind != "array":
        raise ValueError("The tape is malformed and leaves no result.")
    return _WalkResult(root, hides)


def _pop_array(stack: list[Summary]) -> Summary:
    value = _pop(stack)
    if value.kind != "array":
        raise ValueError(STACK_UNDERFLOW)
    return value


def _pop_mask(stack: list[Summary]) -> Summary:
    value = _pop(stack)
    if value.kind != "mask":
        raise ValueError(STACK_UNDERFLOW)
    return value


def _pop(stack: list[Summary]) -> Summary:
    if not stack:
        raise ValueError(STACK_UNDERFLOW)
    return stack.pop()

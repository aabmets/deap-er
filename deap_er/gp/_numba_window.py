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
import math
from typing import Any

from ._numba_codes import (
    _DELAY,
    _DIFF,
    _EMA,
    _ROLL_MAX,
    _ROLL_MEAN,
    _ROLL_MIN,
    _ROLL_STD,
    _ROLL_SUM,
    _UNKNOWN_OPCODE,
)

__all__: list[str] = []


def _apply_window(  # pragma: no cover
    op: int, rows: int, sp: int, stack: Any, scratch: Any, arg: int
) -> int:
    """Apply a delay, difference, rolling, or EMA opcode.

    Args:
        op: Opcode in the window group.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        scratch: Spare row of ``rows`` values.
        arg: Window length.

    Returns:
        The updated stack pointer.

    Raises:
        ValueError: If ``op`` is not a window opcode.
    """
    if op in (_DELAY, _DIFF):
        _apply_shift(op, rows, sp, stack, arg)
        return sp
    if op in (_ROLL_SUM, _ROLL_MEAN, _ROLL_STD):
        _roll_stats(op, rows, sp, stack, scratch, arg)
        return sp
    if op in (_ROLL_MIN, _ROLL_MAX):
        _roll_minmax(op, rows, sp, stack, scratch, arg)
        return sp
    if op == _EMA:
        _roll_ema(rows, sp, stack, scratch, arg)
        return sp
    raise ValueError(_UNKNOWN_OPCODE)


def _apply_shift(op: int, rows: int, sp: int, stack: Any, arg: int) -> None:  # pragma: no cover
    """Apply a causal delay or difference.

    Args:
        op: ``DELAY`` or ``DIFF``.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        arg: Shift length.
    """
    if op == _DELAY:
        for t in range(rows - 1, -1, -1):
            stack[sp - 1, t] = stack[sp - 1, t - arg] if t >= arg else math.nan
    else:
        for t in range(rows - 1, -1, -1):
            if t >= arg:
                stack[sp - 1, t] = stack[sp - 1, t] - stack[sp - 1, t - arg]
            else:
                stack[sp - 1, t] = math.nan


def _roll_stats(  # pragma: no cover
    op: int, rows: int, sp: int, stack: Any, scratch: Any, arg: int
) -> None:
    """Write a rolling sum, mean, or population standard deviation.

    Args:
        op: One of the rolling reduction opcodes.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        scratch: Spare row of ``rows`` values.
        arg: Window length.
    """
    for t in range(rows):
        if t + 1 < arg:
            scratch[t] = math.nan
            continue
        total = 0.0
        squares = 0.0
        for j in range(t - arg + 1, t + 1):
            value = stack[sp - 1, j]
            total += value
            squares += value * value
        scratch[t] = _reduce_stats(op, total, squares, arg)
    for t in range(rows):
        stack[sp - 1, t] = scratch[t]


def _reduce_stats(op: int, total: float, squares: float, arg: int) -> float:  # pragma: no cover
    """Reduce one full window to a sum, mean, or standard deviation.

    Args:
        op: Rolling opcode.
        total: Sum of the window.
        squares: Sum of squares of the window.
        arg: Window length.

    Returns:
        The reduced value.
    """
    if op == _ROLL_SUM:
        return total
    if op == _ROLL_MEAN:
        return total / arg
    mean = total / arg
    variance = squares / arg - mean * mean
    if variance < 0.0:
        variance = 0.0
    return math.sqrt(variance)


def _roll_minmax(  # pragma: no cover
    op: int, rows: int, sp: int, stack: Any, scratch: Any, arg: int
) -> None:
    """Write a rolling minimum or maximum.

    Args:
        op: ``ROLL_MIN`` or ``ROLL_MAX``.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        scratch: Spare row of ``rows`` values.
        arg: Window length.
    """
    for t in range(rows):
        if t + 1 < arg:
            scratch[t] = math.nan
            continue
        scratch[t] = _window_extreme(op, stack, sp, t - arg + 1, t + 1)
    for t in range(rows):
        stack[sp - 1, t] = scratch[t]


def _window_extreme(  # pragma: no cover
    op: int, stack: Any, sp: int, begin: int, end: int
) -> float:
    """Return the min or max of ``stack[sp - 1, begin:end]``.

    A ``nan`` in the window makes the result ``nan``.

    Args:
        op: ``ROLL_MIN`` or ``ROLL_MAX``.
        stack: Column-length workspace.
        sp: Current stack pointer.
        begin: Inclusive start index.
        end: Exclusive stop index.

    Returns:
        The extreme value, or ``nan``.
    """
    best = stack[sp - 1, begin]
    for j in range(begin + 1, end):
        value = stack[sp - 1, j]
        if math.isnan(value) or math.isnan(best):
            best = math.nan
        elif op == _ROLL_MIN:
            if value < best:
                best = value
        elif value > best:
            best = value
    return float(best)


def _roll_ema(rows: int, sp: int, stack: Any, scratch: Any, arg: int) -> None:  # pragma: no cover
    """Write a causal exponential moving average.

    Args:
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        scratch: Spare row of ``rows`` values.
        arg: Span of the average.
    """
    start = -1
    for t in range(rows):
        if math.isfinite(stack[sp - 1, t]):
            start = t
            break
    if start < 0 or arg > rows:
        for t in range(rows):
            scratch[t] = math.nan
    else:
        _fill_ema(rows, sp, stack, scratch, arg, start)
    for t in range(rows):
        stack[sp - 1, t] = scratch[t]


def _fill_ema(  # pragma: no cover
    rows: int, sp: int, stack: Any, scratch: Any, arg: int, start: int
) -> None:
    """Fill ``scratch`` with the EMA of ``stack[sp - 1]`` from ``start``.

    Args:
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        scratch: Spare row of ``rows`` values.
        arg: Span of the average.
        start: Index of the first finite sample.
    """
    alpha = 2.0 / (arg + 1.0)
    previous = stack[sp - 1, start]
    for t in range(start):
        scratch[t] = math.nan
    scratch[start] = previous
    for t in range(start + 1, rows):
        previous = alpha * stack[sp - 1, t] + (1.0 - alpha) * previous
        scratch[t] = previous
    stop = start + arg - 1
    if stop > rows:
        stop = rows
    for t in range(stop):
        scratch[t] = math.nan

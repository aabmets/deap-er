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

from . import numba_codes as codes
from .numba_window_extreme import roll_minmax
from .numba_window_roll import roll_stats

__all__: list[str] = ["apply_window"]


def apply_window(  # pragma: no cover
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
    if op in (codes.DELAY, codes.DIFF):
        apply_shift(op, rows, sp, stack, arg)
        return sp
    if op in (codes.ROLL_SUM, codes.ROLL_MEAN, codes.ROLL_STD):
        roll_stats(op, rows, sp, stack, scratch, arg)
        return sp
    if op in (codes.ROLL_MIN, codes.ROLL_MAX):
        roll_minmax(op, rows, sp, stack, scratch, arg)
        return sp
    if op == codes.EMA:
        roll_ema(rows, sp, stack, scratch, arg)
        return sp
    raise ValueError(codes.UNKNOWN_OPCODE)


def apply_shift(op: int, rows: int, sp: int, stack: Any, arg: int) -> None:  # pragma: no cover
    """Apply a causal delay or difference.

    Args:
        op: ``DELAY`` or ``DIFF``.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        arg: Shift length.
    """
    if op == codes.DELAY:
        for t in range(rows - 1, -1, -1):
            stack[sp - 1, t] = stack[sp - 1, t - arg] if t >= arg else math.nan
    else:
        for t in range(rows - 1, -1, -1):
            if t >= arg:
                stack[sp - 1, t] = stack[sp - 1, t] - stack[sp - 1, t - arg]
            else:
                stack[sp - 1, t] = math.nan


def roll_ema(rows: int, sp: int, stack: Any, scratch: Any, arg: int) -> None:  # pragma: no cover
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
        fill_ema(rows, sp, stack, scratch, arg, start)
    for t in range(rows):
        stack[sp - 1, t] = scratch[t]


def fill_ema(  # pragma: no cover
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

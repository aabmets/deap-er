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
from .numba_window_extreme import roll_extreme

__all__: list[str] = ["apply_ts_window"]


def apply_ts_window(  # pragma: no cover
    op: int, rows: int, sp: int, stack: Any, scratch: Any, arg: int
) -> int:
    """Apply a time-series rank or arg-extremum opcode.

    Args:
        op: Opcode in the time-series window group.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        scratch: Spare row of ``rows`` values.
        arg: Window length.

    Returns:
        The updated stack pointer.

    Raises:
        ValueError: If ``op`` is not a time-series window opcode.
    """
    if op not in (codes.TS_RANK, codes.TS_ARGMAX, codes.TS_ARGMIN):
        raise ValueError(codes.UNKNOWN_OPCODE)
    roll_ts_window(op, rows, sp, stack, scratch, arg)
    return sp


def roll_ts_window(  # pragma: no cover
    op: int, rows: int, sp: int, stack: Any, scratch: Any, arg: int
) -> None:
    """Write a time-series rank or arg-extremum into scratch, then back.

    Args:
        op: Time-series window opcode.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        scratch: Spare row of ``rows`` values.
        arg: Window length.
    """
    if op != codes.TS_RANK:
        want_max = 1 if op == codes.TS_ARGMAX else 0
        roll_extreme(want_max, rows, sp, stack, scratch, arg, 1)
        return
    for t in range(rows):
        if t + 1 < arg:
            scratch[t] = math.nan
            continue
        scratch[t] = window_rank(stack, sp, t, arg)
    for t in range(rows):
        stack[sp - 1, t] = scratch[t]


def window_rank(stack: Any, sp: int, t: int, arg: int) -> float:  # pragma: no cover
    """Return the scaled average rank of ``stack[sp - 1, t]``.

    Args:
        stack: Column-length workspace.
        sp: Current stack pointer.
        t: Index of the current sample.
        arg: Window length.

    Returns:
        The percentile rank, or ``nan``.
    """
    if arg == 1:
        return math.nan
    current = stack[sp - 1, t]
    less = 0.0
    equal = 0.0
    for j in range(t - arg + 1, t + 1):
        value = stack[sp - 1, j]
        if math.isnan(value):
            return math.nan
        if value < current:
            less += 1.0
        elif value <= current:
            equal += 1.0
    rank = less + (equal + 1.0) / 2.0
    return (rank - 1.0) / (arg - 1.0)

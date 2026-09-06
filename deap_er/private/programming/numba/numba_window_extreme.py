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

__all__: list[str] = ["roll_extreme", "roll_minmax"]


def roll_extreme(  # pragma: no cover
    want_max: int, rows: int, sp: int, stack: Any, scratch: Any, arg: int, as_age: int
) -> None:
    """Write a rolling min, max, or extremum age with a monotonic ring.

    The free row ``stack[sp]`` holds at most ``arg`` candidate indices.

    Args:
        want_max: ``1`` for a maximum, ``0`` for a minimum.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        scratch: Spare row of ``rows`` values.
        arg: Window length.
        as_age: ``1`` to write how many samples ago the extreme
            occurred, ``0`` to write the extreme value.
    """
    if arg > rows:
        for t in range(rows):
            scratch[t] = math.nan
            stack[sp - 1, t] = math.nan
        return
    head = 0
    tail = 0
    count = 0
    nan_count = 0
    for t in range(rows):
        if t >= arg:
            if math.isnan(stack[sp - 1, t - arg]):
                nan_count -= 1
            while count > 0 and int(stack[sp, head]) <= t - arg:
                head += 1
                if head == arg:
                    head = 0
                count -= 1
        incoming = stack[sp - 1, t]
        if math.isnan(incoming):
            nan_count += 1
        else:
            while count > 0:
                back = tail - 1
                if back < 0:
                    back += arg
                back_value = stack[sp - 1, int(stack[sp, back])]
                dominated = incoming >= back_value if want_max == 1 else incoming <= back_value
                if not dominated:
                    break
                tail = back
                count -= 1
            stack[sp, tail] = float(t)
            tail += 1
            if tail == arg:
                tail = 0
            count += 1
        if t + 1 < arg or nan_count > 0 or count == 0:
            scratch[t] = math.nan
            continue
        front = int(stack[sp, head])
        if as_age == 1:
            scratch[t] = float(t - front)
        else:
            scratch[t] = stack[sp - 1, front]
    for t in range(rows):
        stack[sp - 1, t] = scratch[t]


def roll_minmax(  # pragma: no cover
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
    want_max = 1 if op == codes.ROLL_MAX else 0
    roll_extreme(want_max, rows, sp, stack, scratch, arg, 0)

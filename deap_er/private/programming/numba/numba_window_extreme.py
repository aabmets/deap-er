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


def fill_nan_rows(stack: Any, scratch: Any, sp: int, rows: int) -> None:  # pragma: no cover
    """Fill scratch and the live row with ``nan``.

    Args:
        stack: Column-length workspace.
        scratch: Spare row of ``rows`` values.
        sp: Current stack pointer.
        rows: Number of samples.
    """
    for t in range(rows):
        scratch[t] = math.nan
        stack[sp - 1, t] = math.nan


def drop_outgoing(
    stack: Any, sp: int, head: int, count: int, nan_count: int, t: int, arg: int
) -> tuple[int, int, int]:  # pragma: no cover
    """Remove the sample that just left the window.

    Args:
        stack: Column-length workspace.
        sp: Current stack pointer.
        head: Oldest candidate index.
        count: Number of live candidates.
        nan_count: ``nan`` samples still in the window.
        t: Current sample index.
        arg: Window length.

    Returns:
        The updated head, count, and nan_count.
    """
    if math.isnan(stack[sp - 1, t - arg]):
        nan_count -= 1
    while count > 0 and int(stack[sp, head]) <= t - arg:
        head += 1
        if head == arg:
            head = 0
        count -= 1
    return head, count, nan_count


def pop_dominated(
    stack: Any, sp: int, tail: int, count: int, arg: int, incoming: float, want_max: int
) -> tuple[int, int]:  # pragma: no cover
    """Drop tail candidates that ``incoming`` dominates.

    Args:
        stack: Column-length workspace.
        sp: Current stack pointer.
        tail: Next free ring slot.
        count: Number of live candidates.
        arg: Ring capacity.
        incoming: New finite sample.
        want_max: ``1`` for a maximum, ``0`` for a minimum.

    Returns:
        The updated tail and count.
    """
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
    return tail, count


def ingest_sample(
    stack: Any,
    sp: int,
    tail: int,
    count: int,
    nan_count: int,
    t: int,
    arg: int,
    incoming: float,
    want_max: int,
) -> tuple[int, int, int]:  # pragma: no cover
    """Absorb the sample at ``t`` into the ring state.

    Args:
        stack: Column-length workspace.
        sp: Current stack pointer.
        tail: Next free ring slot.
        count: Number of live candidates.
        nan_count: ``nan`` samples still in the window.
        t: Current sample index.
        arg: Ring capacity.
        incoming: Sample at ``t``.
        want_max: ``1`` for a maximum, ``0`` for a minimum.

    Returns:
        The updated tail, count, and nan_count.
    """
    if math.isnan(incoming):
        return tail, count, nan_count + 1
    tail, count = pop_dominated(stack, sp, tail, count, arg, incoming, want_max)
    stack[sp, tail] = float(t)
    tail += 1
    if tail == arg:
        tail = 0
    return tail, count + 1, nan_count


def write_extreme(
    scratch: Any,
    stack: Any,
    sp: int,
    t: int,
    arg: int,
    nan_count: int,
    count: int,
    head: int,
    as_age: int,
) -> None:  # pragma: no cover
    """Write the current window extreme, or ``nan`` if it is not ready.

    Args:
        scratch: Spare row of values.
        stack: Column-length workspace.
        sp: Current stack pointer.
        t: Current sample index.
        arg: Window length.
        nan_count: ``nan`` samples still in the window.
        count: Number of live candidates.
        head: Oldest candidate index.
        as_age: ``1`` to write the extremum age, ``0`` to write the value.
    """
    if t + 1 < arg or nan_count > 0 or count == 0:
        scratch[t] = math.nan
        return
    front = int(stack[sp, head])
    if as_age == 1:
        scratch[t] = float(t - front)
        return
    scratch[t] = stack[sp - 1, front]


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
        fill_nan_rows(stack, scratch, sp, rows)
        return
    head = 0
    tail = 0
    count = 0
    nan_count = 0
    for t in range(rows):
        if t >= arg:
            head, count, nan_count = drop_outgoing(stack, sp, head, count, nan_count, t, arg)
        incoming = stack[sp - 1, t]
        tail, count, nan_count = ingest_sample(
            stack, sp, tail, count, nan_count, t, arg, incoming, want_max
        )
        write_extreme(scratch, stack, sp, t, arg, nan_count, count, head, as_age)
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

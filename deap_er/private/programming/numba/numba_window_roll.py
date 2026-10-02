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

import numpy

from . import numba_codes as codes
from .numba_window_sum import BLOCK, block_total, fill_lanes, plan_split, window_total

__all__: list[str] = ["roll_stats", "window_lanes"]


def window_lanes(  # pragma: no cover
    rows: int, sp: int, stack: Any, scratch: Any, arg: int
) -> tuple[Any, Any, Any]:
    """Fill the lane sums that every window of ``arg`` samples needs.

    Each block length in the window's split (see ``plan_split``) gets
    a row of lane sums. A single row lives in ``scratch``.

    Args:
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        scratch: Spare row of ``rows`` values.
        arg: Window length, from 8 to ``rows``.

    Returns:
        The lane sums, the row of them for each number of samples per
        lane, and the split plan.
    """
    plan = plan_split(arg)
    used = numpy.zeros(BLOCK // 8 + 1, dtype=numpy.bool_)
    for step in plan:
        if step > 0:
            used[step // 8] = True
    slots = numpy.zeros(used.size, dtype=numpy.int64)
    count = 0
    for terms in range(used.size):
        if used[terms]:
            slots[terms] = count
            count += 1
    if count == 1:
        lanes = scratch.reshape((1, rows))
    else:
        lanes = numpy.empty((count, rows), dtype=numpy.float64)
    for terms in range(used.size):
        if used[terms]:
            fill_lanes(stack, sp - 1, rows, terms, lanes, slots[terms])
    return lanes, slots, plan


def roll_stats(  # pragma: no cover
    op: int, rows: int, sp: int, stack: Any, scratch: Any, arg: int
) -> None:
    """Write a rolling sum or mean.

    Every window is summed from its own samples, in the pairwise order
    of NumPy's ``add.reduce`` (see ``fill_lanes``), so an output
    depends only on the samples in its window and matches the opcode
    backend bit for bit, ``nan`` and infinities included. The first
    ``arg - 1`` samples are ``nan``. The standard deviation is
    ``roll_std``.

    Windows are written from the last one back, in place: a window
    reads samples up to its own index only.

    Args:
        op: ``ROLL_SUM`` or ``ROLL_MEAN``.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        scratch: Spare row of ``rows`` values.
        arg: Window length.
    """
    series = stack[sp - 1]
    last = arg - 2
    if arg < 8:
        for t in range(rows - 1, last, -1):
            total = 0.0
            for index in range(t - arg + 1, t + 1):
                total += series[index]
            series[t] = total
    elif arg <= rows:
        lanes, slots, plan = window_lanes(rows, sp, stack, scratch, arg)
        if arg <= BLOCK:
            for t in range(rows - 1, last, -1):
                series[t] = 0.0 + block_total(series, lanes, 0, t - arg + 1, arg)
        else:
            values = numpy.empty(plan.size, dtype=numpy.float64)
            for t in range(rows - 1, last, -1):
                begin = t - arg + 1
                series[t] = 0.0 + window_total(series, lanes, slots, plan, values, begin)
    if op == codes.ROLL_MEAN:
        for t in range(rows - 1, last, -1):
            series[t] /= arg
    for t in range(min(arg - 1, rows)):
        series[t] = math.nan

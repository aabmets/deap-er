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
from typing import Any

import numpy

__all__: list[str] = ["BLOCK", "fill_lanes", "plan_split", "block_total", "window_total"]

# NumPy's pairwise ``add`` keeps eight interleaved partial sums over a
# block of 8 to ``BLOCK`` samples, then adds the samples past the last
# multiple of 8 in turn. A longer run splits in half, on a multiple of
# 8, until each part is a block.
BLOCK = 128


def fill_lanes(  # pragma: no cover
    stack: Any, row: int, rows: int, terms: int, lanes: Any, slot: int
) -> None:
    """Write every lane partial sum of ``terms`` samples.

    ``lanes[slot, c]`` becomes ``x[c] + x[c + 8] + ...`` over ``terms``
    samples, added left to right. That is one of the eight partial sums
    NumPy keeps for a block starting at ``c - lane``, so the eight
    windows that share a lane share its sum. Each element is added in
    the same order as in NumPy, and the loop runs over contiguous
    samples, so it vectorizes without reordering any addition.

    Args:
        stack: Column-length workspace.
        row: Stack row that holds the series.
        rows: Number of samples.
        terms: Samples per lane.
        lanes: Workspace of shape ``(slots, rows)``.
        slot: Row of ``lanes`` to fill.
    """
    series = stack[row]
    lane = lanes[slot]
    span = rows - 8 * (terms - 1)
    for start in range(span):
        lane[start] = series[start]
    for term in range(1, terms):
        # A view per term lets the loop vectorize; indexing ``series``
        # at ``start + offset`` does not.
        shifted = series[8 * term : 8 * term + span]
        for start in range(span):
            lane[start] += shifted[start]


def plan_split(count: int) -> Any:  # pragma: no cover
    """Lay out NumPy's pairwise split of a run into blocks.

    The plan lists the split in postfix order: a positive entry is a
    block of that many samples, the next one along the run, and ``0``
    adds the two values before it. A run of up to ``BLOCK`` samples is
    one block. The plan is built without recursion, because Numba's
    disk cache cannot reload a recursive function.

    Args:
        count: Run length, at least 8.

    Returns:
        The plan, ``-1`` past its end.
    """
    size = count // 32 + 8
    plan = numpy.full(size, -1, dtype=numpy.int64)
    tasks = numpy.empty(size, dtype=numpy.int64)
    tasks[0] = count
    pending = 1
    written = 0
    while pending > 0:
        pending -= 1
        task = tasks[pending]
        if task <= BLOCK:
            plan[written] = task
            written += 1
            continue
        half = task // 2
        half -= half % 8
        tasks[pending] = 0
        tasks[pending + 1] = task - half
        tasks[pending + 2] = half
        pending += 3
    return plan


def block_total(  # pragma: no cover
    series: Any, lanes: Any, slot: int, begin: int, count: int
) -> float:
    """Sum one block of 8 to ``BLOCK`` samples in NumPy's order.

    The block's eight partial sums come from ``lanes`` and are added in
    NumPy's tree order, then the last ``count % 8`` samples are added
    in turn.

    Args:
        series: Samples.
        lanes: Lane sums (see ``fill_lanes``).
        slot: Row of ``lanes`` for the block's length.
        begin: Inclusive start index.
        count: Number of samples.

    Returns:
        The sum of the block.
    """
    total = (
        (lanes[slot, begin] + lanes[slot, begin + 1])
        + (lanes[slot, begin + 2] + lanes[slot, begin + 3])
    ) + (
        (lanes[slot, begin + 4] + lanes[slot, begin + 5])
        + (lanes[slot, begin + 6] + lanes[slot, begin + 7])
    )
    for index in range(begin + count - count % 8, begin + count):
        total += series[index]
    return float(total)


def window_total(  # pragma: no cover
    series: Any, lanes: Any, slots: Any, plan: Any, values: Any, begin: int
) -> float:
    """Sum one window of more than ``BLOCK`` samples in NumPy's order.

    Each block is summed as in ``block_total``, written out here
    because a call per block costs about twice the whole sum, and the
    block sums are added as the plan says.

    Args:
        series: Samples.
        lanes: Lane sums for every block length the plan uses.
        slots: Row of ``lanes`` for each number of samples per lane.
        plan: Postfix split of the run (see ``plan_split``).
        values: Spare buffer as long as ``plan``.
        begin: Inclusive start index of the window.

    Returns:
        The sum of the window, before NumPy's final ``0.0 +``.
    """
    depth = 0
    offset = begin
    for step in range(plan.size):
        count = plan[step]
        if count < 0:
            break
        if count == 0:
            depth -= 1
            values[depth - 1] = values[depth - 1] + values[depth]
            continue
        slot = slots[count // 8]
        total = (
            (lanes[slot, offset] + lanes[slot, offset + 1])
            + (lanes[slot, offset + 2] + lanes[slot, offset + 3])
        ) + (
            (lanes[slot, offset + 4] + lanes[slot, offset + 5])
            + (lanes[slot, offset + 6] + lanes[slot, offset + 7])
        )
        for index in range(offset + count - count % 8, offset + count):
            total += series[index]
        values[depth] = total
        offset += count
        depth += 1
    return float(values[0])

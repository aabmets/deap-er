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

from .numba_window_pair_reduce import reduce_pair
from .numba_window_scan import (
    EPSILON,
    row_offset,
    scan_pair_moments,
    scan_pair_sums,
    variance_untrusted,
)

__all__: list[str] = ["roll_pair_stats"]


def absorb_pair(  # pragma: no cover
    left: float,
    right: float,
    sign: int,
    nan_count: int,
    inf_count: int,
    total_x: float,
    total_y: float,
    squares_x: float,
    squares_y: float,
    products: float,
    err_x: float,
    err_y: float,
) -> tuple[int, int, float, float, float, float, float, float, float]:
    """Add or remove one shifted pair sample from the moment state.

    Args:
        left: Shifted left-series sample.
        right: Shifted right-series sample.
        sign: ``1`` to add, ``-1`` to remove.
        nan_count: Pairs where either side is ``nan``.
        inf_count: Finite-``nan``-free pairs that still hold an infinity.
        total_x: Sum of finite left samples.
        total_y: Sum of finite right samples.
        squares_x: Sum of squares of finite left samples.
        squares_y: Sum of squares of finite right samples.
        products: Sum of products of finite pairs.
        err_x: Accumulated bound on the error of ``squares_x``.
        err_y: Accumulated bound on the error of ``squares_y``.

    Returns:
        The updated counts, finite-only accumulators, and error bounds.
    """
    if math.isnan(left) or math.isnan(right):
        return (
            nan_count + sign,
            inf_count,
            total_x,
            total_y,
            squares_x,
            squares_y,
            products,
            err_x,
            err_y,
        )
    if not math.isfinite(left) or not math.isfinite(right):
        return (
            nan_count,
            inf_count + sign,
            total_x,
            total_y,
            squares_x,
            squares_y,
            products,
            err_x,
            err_y,
        )
    square_x = left * left
    square_y = right * right
    return (
        nan_count,
        inf_count,
        total_x + sign * left,
        total_y + sign * right,
        squares_x + sign * square_x,
        squares_y + sign * square_y,
        products + sign * left * right,
        err_x + EPSILON * square_x,
        err_y + EPSILON * square_y,
    )


def roll_pair_stats(  # pragma: no cover
    op: int, rows: int, sp: int, stack: Any, scratch: Any, arg: int
) -> None:
    """Write a rolling pair statistic into scratch, then onto the left row.

    Both series are shifted by ``row_offset``, and a window is
    recentered on its own means whenever a running variance has lost
    too many digits. A rebuild also takes its shifts from the window
    it rebuilds, so series that wander stay on the running path.

    Args:
        op: Pair-window opcode.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace. Left is ``sp - 2``, right is
            ``sp - 1``.
        scratch: Spare row of ``rows`` values.
        arg: Window length.
    """
    if arg > rows:
        for t in range(rows):
            scratch[t] = math.nan
            stack[sp - 2, t] = math.nan
        return
    off_x = row_offset(stack, sp - 2, 0, rows)
    off_y = row_offset(stack, sp - 1, 0, rows)
    nan_count = 0
    inf_count = 0
    total_x = 0.0
    total_y = 0.0
    squares_x = 0.0
    squares_y = 0.0
    products = 0.0
    err_x = 0.0
    err_y = 0.0
    for t in range(rows):
        if t >= arg:
            (
                nan_count,
                inf_count,
                total_x,
                total_y,
                squares_x,
                squares_y,
                products,
                err_x,
                err_y,
            ) = absorb_pair(
                stack[sp - 2, t - arg] - off_x,
                stack[sp - 1, t - arg] - off_y,
                -1,
                nan_count,
                inf_count,
                total_x,
                total_y,
                squares_x,
                squares_y,
                products,
                err_x,
                err_y,
            )
        (
            nan_count,
            inf_count,
            total_x,
            total_y,
            squares_x,
            squares_y,
            products,
            err_x,
            err_y,
        ) = absorb_pair(
            stack[sp - 2, t] - off_x,
            stack[sp - 1, t] - off_y,
            1,
            nan_count,
            inf_count,
            total_x,
            total_y,
            squares_x,
            squares_y,
            products,
            err_x,
            err_y,
        )
        if t + 1 < arg or nan_count > 0:
            scratch[t] = math.nan
            continue
        mean_x = total_x / arg
        mean_y = total_y / arg
        cov = products / arg - mean_x * mean_y
        var_x = squares_x / arg - mean_x * mean_x
        var_y = squares_y / arg - mean_y * mean_y
        recenter = variance_untrusted(var_x, err_x, arg) or variance_untrusted(var_y, err_y, arg)
        if inf_count > 0:
            cov, var_x, var_y = scan_pair_moments(stack, sp - 2, sp - 1, t - arg + 1, t + 1)
        elif recenter:
            off_x = row_offset(stack, sp - 2, t - arg + 1, t + 1)
            off_y = row_offset(stack, sp - 1, t - arg + 1, t + 1)
            total_x, total_y, squares_x, squares_y, products = scan_pair_sums(
                stack, sp - 2, sp - 1, t - arg + 1, t + 1, off_x, off_y
            )
            err_x = EPSILON * squares_x
            err_y = EPSILON * squares_y
            cov, var_x, var_y = scan_pair_moments(stack, sp - 2, sp - 1, t - arg + 1, t + 1)
        scratch[t] = reduce_pair(op, cov, var_x, var_y)
    for t in range(rows):
        stack[sp - 2, t] = scratch[t]

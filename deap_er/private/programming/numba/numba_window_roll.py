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
from .numba_window_scan import (
    EPSILON,
    add_compensated,
    row_offset,
)
from .numba_window_std import std_window

__all__: list[str] = ["roll_stats", "reduce_stats", "resync_sums"]


def resync_sums(  # pragma: no cover
    stack: Any,
    row: int,
    t: int,
    arg: int,
    offset: float,
    empty: bool,
    sums: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Keep the running sums from carrying rounding past a window.

    A window with no finite sample has sums of exactly zero, so they
    start over, which keeps an earlier symbol's rounding from leaking
    past NaN padding. A total that overflowed is rescanned from the
    window, so it comes back once the overflow has left.

    Args:
        stack: Column-length workspace.
        row: Stack row that holds the series.
        t: Current sample index.
        arg: Window length.
        offset: Shift subtracted from every sample.
        empty: Whether the window holds no finite sample.
        sums: ``total``, its compensation, ``squares``, and ``err_bound``.

    Returns:
        The resynced ``sums``.
    """
    if empty:
        return 0.0, 0.0, 0.0, 0.0
    total, comp, squares, err_bound = sums
    if math.isfinite(total + comp):
        return sums
    total = 0.0
    comp = 0.0
    for index in range(max(t - arg + 1, 0), t + 1):
        value = float(stack[row, index]) - offset
        if math.isfinite(value):
            total, comp = add_compensated(total, comp, value)
    return total, comp, squares, err_bound


def absorb_stat(  # pragma: no cover
    value: float,
    sign: int,
    counts: tuple[int, int, int],
    total: float,
    comp: float,
    squares: float,
    err_bound: float,
) -> tuple[tuple[int, int, int], float, float, float, float]:
    """Add or remove one shifted sample from the running moment state.

    The sum is compensated, so a sample far larger than the rest of
    the window leaves no rounding behind once it has left.

    Args:
        value: Shifted sample to apply.
        sign: ``1`` to add, ``-1`` to remove.
        counts: Numbers of ``nan``, ``+inf``, and ``-inf`` samples in
            the window.
        total: Sum of the finite samples.
        comp: Compensation term of ``total``.
        squares: Sum of squares of the finite samples.
        err_bound: Accumulated bound on the error of ``squares``.

    Returns:
        The updated ``counts``, ``total``, ``comp``, ``squares``, and
        ``err_bound``.
    """
    nan_count, pos_inf, neg_inf = counts
    if math.isnan(value):
        return (nan_count + sign, pos_inf, neg_inf), total, comp, squares, err_bound
    if math.isinf(value):
        if value > 0.0:
            return (nan_count, pos_inf + sign, neg_inf), total, comp, squares, err_bound
        return (nan_count, pos_inf, neg_inf + sign), total, comp, squares, err_bound
    square = value * value
    total, comp = add_compensated(total, comp, sign * value)
    return counts, total, comp, squares + sign * square, err_bound + EPSILON * square


def window_totals(  # pragma: no cover
    pos_inf: int, neg_inf: int, total: float, squares: float
) -> tuple[float, float]:
    """Rebuild the IEEE window sum and sum of squares.

    Args:
        pos_inf: Number of ``+inf`` samples in the window.
        neg_inf: Number of ``-inf`` samples in the window.
        total: Sum of the finite samples.
        squares: Sum of squares of the finite samples.

    Returns:
        The sum and sum of squares, matching ``add.reduce`` on a
        window that may hold infinities.
    """
    if pos_inf > 0 and neg_inf > 0:
        return math.nan, math.inf
    if pos_inf > 0:
        return math.inf, math.inf
    if neg_inf > 0:
        return -math.inf, math.inf
    return total, squares


def reduce_stats(op: int, total: float, variance: float, arg: int) -> float:  # pragma: no cover
    """Reduce one full window to a sum, mean, or standard deviation.

    Args:
        op: Rolling opcode.
        total: Sum of the window.
        variance: Population variance of the window.
        arg: Window length.

    Returns:
        The reduced value.
    """
    if op == codes.ROLL_SUM:
        return total
    if op == codes.ROLL_MEAN:
        return total / arg
    if variance < 0.0:
        variance = 0.0
    return math.sqrt(variance)


def roll_stats(  # pragma: no cover
    op: int, rows: int, sp: int, stack: Any, scratch: Any, arg: int
) -> None:
    """Write a rolling sum, mean, or population standard deviation.

    Sums and means run on the raw samples with a compensated running
    sum (see ``resync_sums``). A standard deviation runs on samples
    shifted by ``row_offset``, and recenters the window on its own mean
    whenever the running variance has lost too many digits. A rebuild
    also takes its shift from the window it rebuilds, so a series that
    wanders stays on the running path.

    Args:
        op: One of the rolling reduction opcodes.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        scratch: Spare row of ``rows`` values.
        arg: Window length.
    """
    if arg > rows:
        for t in range(rows):
            scratch[t] = math.nan
            stack[sp - 1, t] = math.nan
        return
    offset = row_offset(stack, sp - 1, 0, rows) if op == codes.ROLL_STD else 0.0
    counts = (0, 0, 0)
    total = 0.0
    comp = 0.0
    squares = 0.0
    err_bound = 0.0
    for t in range(rows):
        if t >= arg:
            counts, total, comp, squares, err_bound = absorb_stat(
                stack[sp - 1, t - arg] - offset, -1, counts, total, comp, squares, err_bound
            )
        counts, total, comp, squares, err_bound = absorb_stat(
            stack[sp - 1, t] - offset, 1, counts, total, comp, squares, err_bound
        )
        empty = counts[0] + counts[1] + counts[2] == min(t + 1, arg)
        total, comp, squares, err_bound = resync_sums(
            stack, sp - 1, t, arg, offset, empty, (total, comp, squares, err_bound)
        )
        if t + 1 < arg or counts[0] > 0:
            scratch[t] = math.nan
            continue
        win_total, win_squares = window_totals(counts[1], counts[2], total + comp, squares)
        variance = math.nan
        if op == codes.ROLL_STD:
            variance, offset, total, squares, err_bound = std_window(
                stack,
                sp,
                t,
                arg,
                win_total,
                win_squares,
                counts[1],
                counts[2],
                total + comp,
                squares,
                err_bound,
                offset,
            )
            comp = 0.0
        scratch[t] = reduce_stats(op, win_total, variance, arg)
    for t in range(rows):
        stack[sp - 1, t] = scratch[t]

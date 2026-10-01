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
from .numba_window_scan import add_compensated

__all__: list[str] = ["roll_stats", "resync_sums"]


def resync_sums(  # pragma: no cover
    stack: Any,
    row: int,
    t: int,
    arg: int,
    empty: bool,
    total: float,
    comp: float,
) -> tuple[float, float]:
    """Keep the running sum from carrying rounding past a window.

    A window with no finite sample has a sum of exactly zero, so it
    starts over, which keeps an earlier symbol's rounding from leaking
    past NaN padding. A total that overflowed is rescanned from the
    window, so it comes back once the overflow has left.

    Args:
        stack: Column-length workspace.
        row: Stack row that holds the series.
        t: Current sample index.
        arg: Window length.
        empty: Whether the window holds no finite sample.
        total: Running sum of the finite samples.
        comp: Compensation term of ``total``.

    Returns:
        The resynced ``total`` and ``comp``.
    """
    if empty:
        return 0.0, 0.0
    if math.isfinite(total + comp):
        return total, comp
    total = 0.0
    comp = 0.0
    for index in range(max(t - arg + 1, 0), t + 1):
        value = float(stack[row, index])
        if math.isfinite(value):
            total, comp = add_compensated(total, comp, value)
    return total, comp


def absorb_stat(  # pragma: no cover
    value: float,
    sign: int,
    counts: tuple[int, int, int],
    total: float,
    comp: float,
) -> tuple[tuple[int, int, int], float, float]:
    """Add or remove one sample from the running sum.

    The sum is compensated, so a sample far larger than the rest of
    the window leaves no rounding behind once it has left.

    Args:
        value: Sample to apply.
        sign: ``1`` to add, ``-1`` to remove.
        counts: Numbers of ``nan``, ``+inf``, and ``-inf`` samples in
            the window.
        total: Sum of the finite samples.
        comp: Compensation term of ``total``.

    Returns:
        The updated ``counts``, ``total``, and ``comp``.
    """
    nan_count, pos_inf, neg_inf = counts
    if math.isnan(value):
        return (nan_count + sign, pos_inf, neg_inf), total, comp
    if math.isinf(value):
        if value > 0.0:
            return (nan_count, pos_inf + sign, neg_inf), total, comp
        return (nan_count, pos_inf, neg_inf + sign), total, comp
    total, comp = add_compensated(total, comp, sign * value)
    return counts, total, comp


def window_total(pos_inf: int, neg_inf: int, total: float) -> float:  # pragma: no cover
    """Rebuild the IEEE window sum.

    Args:
        pos_inf: Number of ``+inf`` samples in the window.
        neg_inf: Number of ``-inf`` samples in the window.
        total: Sum of the finite samples.

    Returns:
        The sum, matching ``add.reduce`` on a window that may hold
        infinities.
    """
    if pos_inf > 0 and neg_inf > 0:
        return math.nan
    if pos_inf > 0:
        return math.inf
    if neg_inf > 0:
        return -math.inf
    return total


def roll_stats(  # pragma: no cover
    op: int, rows: int, sp: int, stack: Any, scratch: Any, arg: int
) -> None:
    """Write a rolling sum or mean.

    Both run on the raw samples with a compensated running sum (see
    ``resync_sums``). The standard deviation is ``roll_std``.

    Args:
        op: ``ROLL_SUM`` or ``ROLL_MEAN``.
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
    counts = (0, 0, 0)
    total = 0.0
    comp = 0.0
    for t in range(rows):
        if t >= arg:
            counts, total, comp = absorb_stat(stack[sp - 1, t - arg], -1, counts, total, comp)
        counts, total, comp = absorb_stat(stack[sp - 1, t], 1, counts, total, comp)
        empty = counts[0] + counts[1] + counts[2] == min(t + 1, arg)
        total, comp = resync_sums(stack, sp - 1, t, arg, empty, total, comp)
        if t + 1 < arg or counts[0] > 0:
            scratch[t] = math.nan
            continue
        win_total = window_total(counts[1], counts[2], total + comp)
        scratch[t] = win_total if op == codes.ROLL_SUM else win_total / arg
    for t in range(rows):
        stack[sp - 1, t] = scratch[t]

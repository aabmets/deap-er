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

__all__: list[str] = ["apply_pair_window"]


def apply_pair_window(  # pragma: no cover
    op: int, rows: int, sp: int, stack: Any, scratch: Any, arg: int
) -> int:
    """Apply a two-input rolling covariance, correlation, or beta.

    Args:
        op: Opcode in the pair-window group.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        scratch: Spare row of ``rows`` values.
        arg: Window length.

    Returns:
        The updated stack pointer.

    Raises:
        ValueError: If ``op`` is not a pair-window opcode.
    """
    if op not in (codes.ROLL_CORR, codes.ROLL_COV, codes.ROLL_BETA):
        raise ValueError(codes.UNKNOWN_OPCODE)
    roll_pair_stats(op, rows, sp, stack, scratch, arg)
    return sp - 1


def roll_pair_stats(  # pragma: no cover
    op: int, rows: int, sp: int, stack: Any, scratch: Any, arg: int
) -> None:
    """Write a rolling pair statistic into scratch, then onto the left row.

    Args:
        op: Pair-window opcode.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace. Left is ``sp - 2``, right is
            ``sp - 1``.
        scratch: Spare row of ``rows`` values.
        arg: Window length.
    """
    for t in range(rows):
        if t + 1 < arg:
            scratch[t] = math.nan
            continue
        total_x = 0.0
        total_y = 0.0
        squares_x = 0.0
        squares_y = 0.0
        products = 0.0
        for j in range(t - arg + 1, t + 1):
            left = stack[sp - 2, j]
            right = stack[sp - 1, j]
            total_x += left
            total_y += right
            squares_x += left * left
            squares_y += right * right
            products += left * right
        scratch[t] = reduce_pair(op, total_x, total_y, squares_x, squares_y, products, arg)
    for t in range(rows):
        stack[sp - 2, t] = scratch[t]


def reduce_pair(  # pragma: no cover
    op: int,
    total_x: float,
    total_y: float,
    squares_x: float,
    squares_y: float,
    products: float,
    arg: int,
) -> float:
    """Reduce one full pair window to covariance, correlation, or beta.

    Args:
        op: Pair-window opcode.
        total_x: Sum of the left window.
        total_y: Sum of the right window.
        squares_x: Sum of squares of the left window.
        squares_y: Sum of squares of the right window.
        products: Sum of products of the two windows.
        arg: Window length.

    Returns:
        The reduced value, or ``nan`` when a variance in the
        denominator is zero.
    """
    count = float(arg)
    mean_x = total_x / count
    mean_y = total_y / count
    cov = products / count - mean_x * mean_y
    var_x = squares_x / count - mean_x * mean_x
    var_y = squares_y / count - mean_y * mean_y
    if var_x < 0.0:
        var_x = 0.0
    if var_y < 0.0:
        var_y = 0.0
    if op == codes.ROLL_COV:
        return cov
    if op == codes.ROLL_CORR:
        if var_x <= 0.0 or var_y <= 0.0:
            return math.nan
        return cov / math.sqrt(var_x * var_y)
    if var_y <= 0.0:
        return math.nan
    return cov / var_y

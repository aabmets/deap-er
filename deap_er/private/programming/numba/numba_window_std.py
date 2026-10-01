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

from .numba_window_scan import (
    EPSILON,
    row_offset,
    scan_sums,
    scan_variance,
    variance_untrusted,
)

__all__: list[str] = ["std_window"]


def std_window(  # pragma: no cover
    stack: Any,
    sp: int,
    t: int,
    arg: int,
    win_total: float,
    win_squares: float,
    pos_inf: int,
    neg_inf: int,
    total: float,
    squares: float,
    err_bound: float,
    offset: float,
) -> tuple[float, float, float, float, float]:
    """Rebuild a rolling standard-deviation window when the running path drifts.

    Args:
        stack: Column-length workspace.
        sp: Current stack pointer.
        t: Current sample index.
        arg: Window length.
        win_total: IEEE window sum.
        win_squares: IEEE window sum of squares.
        pos_inf: Count of ``+inf`` samples.
        neg_inf: Count of ``-inf`` samples.
        total: Finite-sample sum.
        squares: Finite-sample sum of squares.
        err_bound: Accumulated error bound on ``squares``.
        offset: Current centering shift.

    Returns:
        Variance, offset, total, squares, and err_bound.
    """
    mean = win_total / arg
    variance = win_squares / arg - mean * mean
    if pos_inf > 0 or neg_inf > 0:
        return scan_variance(stack, sp - 1, t - arg + 1, t + 1), offset, total, squares, err_bound
    if math.isfinite(variance) and not variance_untrusted(variance, err_bound, arg):
        return variance, offset, total, squares, err_bound
    offset = row_offset(stack, sp - 1, t - arg + 1, t + 1)
    total, squares = scan_sums(stack, sp - 1, t - arg + 1, t + 1, offset)
    err_bound = EPSILON * squares
    return scan_variance(stack, sp - 1, t - arg + 1, t + 1), offset, total, squares, err_bound

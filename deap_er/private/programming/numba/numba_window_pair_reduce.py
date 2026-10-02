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

from . import numba_codes as codes

__all__: list[str] = ["reduce_pair"]


def reduce_pair(  # pragma: no cover
    op: int, cov: float, var_x: float, var_y: float, scale_x: float, scale_y: float
) -> float:
    """Reduce one pair window's moments to covariance, correlation, or beta.

    Args:
        op: Pair-window opcode.
        cov: Scaled population covariance of the window.
        var_x: Scaled population variance of the left window.
        var_y: Scaled population variance of the right window.
        scale_x: Scale of the left window.
        scale_y: Scale of the right window.

    Returns:
        The reduced value, or ``nan`` when a variance in the
        denominator is zero.
    """
    if var_x < 0.0:
        var_x = 0.0
    if var_y < 0.0:
        var_y = 0.0
    if op == codes.ROLL_COV:
        return cov * scale_x * scale_y
    if op == codes.ROLL_CORR:
        if var_x <= 0.0 or var_y <= 0.0:
            return math.nan
        return cov / math.sqrt(var_x * var_y)
    if var_y <= 0.0:
        return math.nan
    return cov / var_y * scale_x / scale_y

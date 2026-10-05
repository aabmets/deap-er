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
from .numba_window_scan import scan_pair_moments

__all__: list[str] = ["roll_pair_stats"]


def roll_pair_stats(  # pragma: no cover
    op: int, rows: int, sp: int, stack: Any, scratch: Any, arg: int
) -> None:
    """Write a rolling pair statistic into scratch, then onto the left row.

    Every window is centered on its own means (see
    ``scan_pair_moments``), so an output depends only on the samples in
    its window, as on the opcode backend. A window that holds a ``nan``
    or an infinity is ``nan``, and the first ``arg - 1`` samples are
    ``nan``.

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
        cov, var_x, var_y, scale_x, scale_y = scan_pair_moments(
            stack, sp - 2, sp - 1, t - arg + 1, t + 1
        )
        scratch[t] = reduce_pair(op, cov, var_x, var_y, scale_x, scale_y)
    for t in range(rows):
        stack[sp - 2, t] = scratch[t]

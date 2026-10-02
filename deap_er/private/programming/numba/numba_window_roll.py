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
from .numba_window_scan import scan_sum

__all__: list[str] = ["roll_stats"]


def roll_stats(  # pragma: no cover
    op: int, rows: int, sp: int, stack: Any, scratch: Any, arg: int
) -> None:
    """Write a rolling sum or mean.

    Every window is summed from its own samples (see ``scan_sum``), so
    an output depends only on the samples in its window, as on the
    opcode backend. A window that holds a ``nan`` is ``nan``, and the
    first ``arg - 1`` samples are ``nan``. The standard deviation is
    ``roll_std``.

    Args:
        op: ``ROLL_SUM`` or ``ROLL_MEAN``.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        scratch: Spare row of ``rows`` values.
        arg: Window length.
    """
    for t in range(rows):
        if t + 1 < arg:
            scratch[t] = math.nan
            continue
        total = scan_sum(stack, sp - 1, t - arg + 1, t + 1)
        scratch[t] = total if op == codes.ROLL_SUM else total / arg
    for t in range(rows):
        stack[sp - 1, t] = scratch[t]

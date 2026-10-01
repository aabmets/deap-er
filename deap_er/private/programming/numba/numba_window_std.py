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

from .numba_window_scan import scan_variance

__all__: list[str] = ["roll_std"]


def roll_std(  # pragma: no cover
    rows: int, sp: int, stack: Any, scratch: Any, arg: int
) -> None:
    """Write a rolling population standard deviation.

    Every window is centered on its own mean (see ``scan_variance``),
    so an output depends only on the samples in its window, as on the
    opcode backend. A window that holds a ``nan`` or an infinity is
    ``nan``, and the first ``arg - 1`` samples are ``nan``.

    Args:
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
        variance = scan_variance(stack, sp - 1, t - arg + 1, t + 1)
        if variance < 0.0:
            variance = 0.0
        scratch[t] = math.sqrt(variance)
    for t in range(rows):
        stack[sp - 1, t] = scratch[t]

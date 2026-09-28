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
from __future__ import annotations

import math

from .opcode_set import Opcode
from .tape_interval_arith import hull
from .tape_interval_ops import Summary, both_can_be_finite, const_pair
from .tape_lookback import opcode_lookback

__all__: list[str] = ["apply_pair_window", "apply_window"]

TS_OUTPUT = frozenset({int(Opcode.TS_RANK), int(Opcode.TS_ARGMAX), int(Opcode.TS_ARGMIN)})
_SHIFTS = frozenset({int(Opcode.DELAY), int(Opcode.DIFF), int(Opcode.EMA)})


def apply_window(opcode: int, child: Summary, operand: int) -> Summary:
    """Propagate one causal window opcode.

    Args:
        opcode: Window instruction.
        child: Summary of the operand.
        operand: Window length folded into the instruction.

    Returns:
        Summary of the result.
    """
    # ema has no finite lookback; its per-segment warmup of window - 1 still
    # bounds first_finite, which is all interval analysis needs from it.
    extra = max(operand - 1, 0) if opcode == int(Opcode.EMA) else opcode_lookback(opcode, operand)
    lookback = child.lookback + extra
    warmup = extra if opcode in _SHIFTS else max(extra - 1, 0)
    first_finite = child.first_finite + warmup
    if opcode in TS_OUTPUT:
        if opcode == int(Opcode.TS_RANK):
            lo, hi = 0.0, 1.0
        else:
            lo, hi = 0.0, float(max(operand - 1, 0))
        return Summary(lo, hi, lookback, first_finite, False, child.can_finite, "array")
    lo, hi = _window_range(opcode, child, operand)
    return Summary(lo, hi, lookback, first_finite, child.const, child.can_finite, "array")


def _window_range(opcode: int, child: Summary, operand: int) -> tuple[float, float]:
    """Return the envelope of a single-series window over ``child``."""
    if opcode == int(Opcode.DIFF):
        return hull((child.lo - child.hi, child.hi - child.lo))
    if opcode == int(Opcode.ROLL_SUM):
        return hull((operand * child.lo, operand * child.hi))
    if opcode == int(Opcode.ROLL_STD):
        # A population deviation never exceeds half the sample range.
        return hull((0.0, (child.hi - child.lo) / 2.0))
    # delay, mean, min, max, and ema stay inside the input hull.
    return child.lo, child.hi


def apply_pair_window(opcode: int, left: Summary, right: Summary, operand: int) -> Summary:
    """Propagate one pair-window opcode.

    Args:
        opcode: Pair-window instruction.
        left: Summary of the first operand.
        right: Summary of the second operand.
        operand: Window length folded into the instruction.

    Returns:
        Summary of the result.
    """
    extra = opcode_lookback(opcode, operand)
    lookback = max(left.lookback, right.lookback) + extra
    first_finite = max(left.first_finite, right.first_finite) + max(extra - 1, 0)
    if opcode == int(Opcode.ROLL_CORR):
        lo, hi = -1.0, 1.0
    elif opcode == int(Opcode.ROLL_COV):
        # |cov| is at most the product of the two population deviations.
        spread = (left.hi - left.lo) * (right.hi - right.lo) / 4.0
        lo, hi = (-math.inf, math.inf) if math.isnan(spread) else (-spread, spread)
    else:
        # beta divides by a variance that may be arbitrarily small.
        lo, hi = -math.inf, math.inf
    const = const_pair(left, right)
    can_finite = both_can_be_finite(left, right)
    return Summary(lo, hi, lookback, first_finite, const, can_finite, "array")

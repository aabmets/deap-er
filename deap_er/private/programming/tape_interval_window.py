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

from .opcode_set import Opcode
from .tape_interval_ops import Summary
from .tape_lookback import opcode_lookback

__all__: list[str] = ["apply_pair_window", "apply_window"]

TS_OUTPUT = frozenset({int(Opcode.TS_RANK), int(Opcode.TS_ARGMAX), int(Opcode.TS_ARGMIN)})


def apply_window(opcode: int, child: Summary, operand: int) -> Summary:
    """Propagate one causal window opcode."""
    extra = opcode_lookback(opcode, operand)
    lookback = child.lookback + extra
    warmup = extra if opcode in {int(Opcode.DELAY), int(Opcode.DIFF)} else max(extra - 1, 0)
    first_finite = child.first_finite + warmup
    if opcode in TS_OUTPUT:
        if opcode == int(Opcode.TS_RANK):
            lo, hi = 0.0, 1.0
        else:
            lo, hi = 0.0, float(max(operand - 1, 0))
        return Summary(lo, hi, lookback, first_finite, False, child.can_finite, "array")
    return Summary(
        child.lo,
        child.hi,
        lookback,
        first_finite,
        child.const,
        child.can_finite,
        "array",
    )


def apply_pair_window(opcode: int, left: Summary, right: Summary, operand: int) -> Summary:
    """Propagate one pair-window opcode."""
    extra = opcode_lookback(opcode, operand)
    lookback = max(left.lookback, right.lookback) + extra
    first_finite = max(left.first_finite, right.first_finite) + max(extra - 1, 0)
    lo = min(left.lo, right.lo)
    hi = max(left.hi, right.hi)
    const = left.const and right.const and left.lo == left.hi and right.lo == right.hi
    can_finite = left.can_finite or right.can_finite
    return Summary(lo, hi, lookback, first_finite, const, can_finite, "array")

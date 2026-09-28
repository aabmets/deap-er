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

from collections.abc import Sequence
from dataclasses import dataclass

import numpy

from .opcode_set import Opcode

__all__: list[str] = [
    "BINARY",
    "BINARY_PROTECTED",
    "COMPARISONS",
    "PAIR_WINDOWED",
    "STACK_UNDERFLOW",
    "UNARY",
    "WINDOWED",
    "Summary",
    "both_can_be_finite",
    "hides_warmup",
    "merge_arrays",
    "normalize_bounds",
]

STACK_UNDERFLOW = "The tape is malformed and underflows the interval stack."
COMPARISONS = frozenset(
    {
        int(Opcode.GT),
        int(Opcode.LT),
        int(Opcode.GE),
        int(Opcode.LE),
        int(Opcode.EQ),
    }
)
UNARY = frozenset(
    {
        int(Opcode.NEG),
        int(Opcode.ABS),
        int(Opcode.LOG),
        int(Opcode.SQRT),
        int(Opcode.SIN),
        int(Opcode.COS),
        int(Opcode.NOT),
    }
)
BINARY = frozenset(
    {
        int(Opcode.ADD),
        int(Opcode.SUB),
        int(Opcode.MUL),
        int(Opcode.GT),
        int(Opcode.LT),
        int(Opcode.GE),
        int(Opcode.LE),
        int(Opcode.EQ),
        int(Opcode.AND),
        int(Opcode.OR),
    }
)
BINARY_PROTECTED = frozenset({int(Opcode.DIV)})
WINDOWED = frozenset(
    {
        int(Opcode.DELAY),
        int(Opcode.DIFF),
        int(Opcode.ROLL_SUM),
        int(Opcode.ROLL_MEAN),
        int(Opcode.ROLL_STD),
        int(Opcode.ROLL_MIN),
        int(Opcode.ROLL_MAX),
        int(Opcode.EMA),
        int(Opcode.TS_RANK),
        int(Opcode.TS_ARGMAX),
        int(Opcode.TS_ARGMIN),
    }
)
PAIR_WINDOWED = frozenset({int(Opcode.ROLL_CORR), int(Opcode.ROLL_COV), int(Opcode.ROLL_BETA)})


@dataclass(frozen=True)
class Summary:
    """Interval summary carried on the static-analysis stack."""

    lo: float
    hi: float
    lookback: int
    first_finite: int
    const: bool
    can_finite: bool
    kind: str
    compared_lookback: int = 0


def normalize_bounds(
    column_bounds: numpy.ndarray | Sequence[tuple[float, float]],
    columns: int,
) -> numpy.ndarray:
    """Validate and coerce per-column ``(low, high)`` bounds.

    Args:
        column_bounds: Caller-supplied bounds for each column.
        columns: Number of columns the tape expects.

    Returns:
        A ``(columns, 2)`` ``float64`` bounds array.

    Raises:
        ValueError: If the bounds shape does not match ``columns``.
    """
    bounds = numpy.asarray(column_bounds, dtype=numpy.float64)
    if bounds.shape != (columns, 2):
        raise ValueError(f"The tape expects {columns} column bounds, got shape {bounds.shape}.")
    return bounds


def hides_warmup(condition: Summary, on_true: Summary, on_false: Summary) -> bool:
    """Return whether ``vwhere`` can mask causal warmup ``nan``."""
    if condition.compared_lookback <= 0:
        return False
    alternate = _warmup_alternate_branch(on_true, on_false)
    if alternate is None:
        return False
    return alternate.can_finite and alternate.first_finite <= 0


def _warmup_alternate_branch(on_true: Summary, on_false: Summary) -> Summary | None:
    if on_true.lookback == 0:
        return on_true
    if on_false.lookback == 0:
        return on_false
    return None


def merge_arrays(left: Summary, right: Summary) -> Summary:
    """Merge branch intervals for ``vwhere``."""
    return Summary(
        min(left.lo, right.lo),
        max(left.hi, right.hi),
        max(left.lookback, right.lookback),
        min(left.first_finite, right.first_finite),
        left.const and right.const and left.lo == left.hi and right.lo == right.hi,
        left.can_finite or right.can_finite,
        "array",
    )


def both_can_be_finite(left: Summary, right: Summary) -> bool:
    """Return whether both operands may be finite on the same row."""
    return left.can_finite and right.can_finite

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
from .tape_lookback import opcode_lookback

__all__: list[str] = [
    "Summary",
    "apply_binary",
    "apply_unary",
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
    alternate = on_true if on_true.lookback == 0 else on_false if on_false.lookback == 0 else None
    if alternate is None:
        return False
    return alternate.can_finite and alternate.first_finite <= 0


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


def apply_unary(opcode: int, child: Summary, fill: float) -> Summary:
    """Propagate one unary opcode."""
    extra = opcode_lookback(opcode, -1)
    lookback = child.lookback + extra
    first_finite = child.first_finite + extra
    if opcode == int(Opcode.NEG):
        lo, hi = -child.hi, -child.lo
        return Summary(lo, hi, lookback, first_finite, child.const, child.can_finite, "array")
    if opcode == int(Opcode.ABS):
        lo = 0.0 if child.lo <= 0.0 <= child.hi else min(abs(child.lo), abs(child.hi))
        hi = max(abs(child.lo), abs(child.hi))
        return Summary(lo, hi, lookback, first_finite, child.const, child.can_finite, "array")
    if opcode == int(Opcode.LOG):
        return protected_unary(child, fill, lookback, first_finite, domain_lo=0.0, strict=True)
    if opcode == int(Opcode.SQRT):
        return protected_unary(child, fill, lookback, first_finite, domain_lo=0.0, strict=False)
    if opcode in {int(Opcode.SIN), int(Opcode.COS)}:
        return Summary(-1.0, 1.0, lookback, first_finite, False, child.can_finite, "array")
    raise ValueError(f"Opcode {opcode} has no interval certificate.")


def protected_unary(
    child: Summary,
    fill: float,
    lookback: int,
    first_finite: int,
    *,
    domain_lo: float,
    strict: bool,
) -> Summary:
    in_domain = child.lo > domain_lo if strict else child.lo >= domain_lo
    if in_domain and (child.hi > domain_lo if strict else child.hi >= domain_lo):
        if strict:
            lo = float(numpy.log(max(child.lo, numpy.finfo(float).tiny)))
            hi = float(numpy.log(child.hi))
        else:
            lo = float(numpy.sqrt(max(child.lo, 0.0)))
            hi = float(numpy.sqrt(child.hi))
        return Summary(lo, hi, lookback, first_finite, child.const, child.can_finite, "array")
    lo = min(fill, child.lo, child.hi)
    hi = max(fill, child.lo, child.hi)
    return Summary(lo, hi, lookback, first_finite, False, child.can_finite, "array")


def apply_binary(opcode: int, left: Summary, right: Summary, fill: float) -> Summary:
    """Propagate one binary opcode."""
    lookback = max(left.lookback, right.lookback)
    first_finite = max(left.first_finite, right.first_finite)
    can_finite = both_can_be_finite(left, right)
    const = left.const and right.const and left.lo == left.hi and right.lo == right.hi
    if opcode == int(Opcode.ADD):
        return Summary(
            left.lo + right.lo,
            left.hi + right.hi,
            lookback,
            first_finite,
            const,
            can_finite,
            "array",
        )
    if opcode == int(Opcode.SUB):
        return Summary(
            left.lo - right.hi,
            left.hi - right.lo,
            lookback,
            first_finite,
            const,
            can_finite,
            "array",
        )
    if opcode == int(Opcode.MUL):
        products = (
            left.lo * right.lo,
            left.lo * right.hi,
            left.hi * right.lo,
            left.hi * right.hi,
        )
        return Summary(
            min(products),
            max(products),
            lookback,
            first_finite,
            const,
            can_finite,
            "array",
        )
    if opcode == int(Opcode.DIV):
        if right.lo <= 0.0 <= right.hi:
            low_quote = left.lo / right.hi if right.hi else fill
            high_quote = left.hi / right.lo if right.lo else fill
            lo = min(low_quote, high_quote, fill)
            hi = max(low_quote, high_quote, fill)
            lo = min(lo, fill, left.lo, left.hi, right.lo, right.hi)
            hi = max(hi, fill, left.lo, left.hi, right.lo, right.hi)
            return Summary(lo, hi, lookback, first_finite, False, can_finite, "array")
        if right.hi < 0.0:
            lo, hi = left.hi / right.lo, left.lo / right.hi
        else:
            lo, hi = left.lo / right.hi, left.hi / right.lo
        return Summary(min(lo, hi), max(lo, hi), lookback, first_finite, const, can_finite, "array")
    raise ValueError(f"Opcode {opcode} has no interval certificate.")

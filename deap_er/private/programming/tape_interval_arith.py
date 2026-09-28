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
from .tape_interval_ops import Summary, both_can_be_finite, const_pair
from .tape_lookback import opcode_lookback

__all__: list[str] = ["apply_binary", "apply_unary", "hull"]

_UNBOUNDED = (-math.inf, math.inf)


def hull(candidates: tuple[float, ...]) -> tuple[float, float]:
    """Return the ``(low, high)`` hull of interval endpoint candidates.

    A ``nan`` candidate comes from ``inf / inf`` or ``inf - inf`` on an
    unbounded operand, where no finite envelope can be claimed.

    Args:
        candidates: Endpoint values to enclose.

    Returns:
        The smallest interval holding every candidate, or the whole
        real line when a candidate is ``nan``.
    """
    if any(math.isnan(value) for value in candidates):
        return _UNBOUNDED
    return min(candidates), max(candidates)


def _product(left: float, right: float) -> float:
    """Multiply interval endpoints, with ``0 * inf`` read as ``0``."""
    if 0.0 in (left, right):
        return 0.0
    return left * right


def apply_unary(opcode: int, child: Summary, fill: float) -> Summary:
    """Propagate one unary opcode.

    Args:
        opcode: Unary instruction.
        child: Summary of the operand.
        fill: Fill of the protected instructions.

    Returns:
        Summary of the result.

    Raises:
        ValueError: If ``opcode`` has no interval rule.
    """
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
        return _protected_unary(child, fill, lookback, first_finite, log=True)
    if opcode == int(Opcode.SQRT):
        return _protected_unary(child, fill, lookback, first_finite, log=False)
    if opcode in {int(Opcode.SIN), int(Opcode.COS)}:
        return Summary(-1.0, 1.0, lookback, first_finite, False, child.can_finite, "array")
    raise ValueError(f"Opcode {opcode} has no interval certificate.")


def _protected_unary(
    child: Summary, fill: float, lookback: int, first_finite: int, *, log: bool
) -> Summary:
    """Propagate a protected ``log`` or ``sqrt`` over its domain and fill."""
    in_domain = child.lo > 0.0 if log else child.lo >= 0.0
    if in_domain:
        if log:
            lo, hi = math.log(child.lo), math.log(child.hi)
        else:
            lo, hi = math.sqrt(child.lo), math.sqrt(child.hi)
        return Summary(lo, hi, lookback, first_finite, child.const, child.can_finite, "array")
    # Part of the operand is out of the domain, where the result is
    # ``fill``. The rest maps onto ``(-inf, log(hi)]`` or ``[0, sqrt(hi)]``.
    lo = hi = fill
    reaches_domain = child.hi > 0.0 if log else child.hi >= 0.0
    if reaches_domain:
        lo = min(fill, -math.inf if log else 0.0)
        hi = max(fill, math.log(child.hi) if log else math.sqrt(child.hi))
    return Summary(lo, hi, lookback, first_finite, False, child.can_finite, "array")


def apply_binary(opcode: int, left: Summary, right: Summary, fill: float) -> Summary:
    """Propagate one arithmetic binary opcode.

    Args:
        opcode: Binary arithmetic instruction.
        left: Summary of the left operand.
        right: Summary of the right operand.
        fill: Fill of the protected instructions.

    Returns:
        Summary of the result.

    Raises:
        ValueError: If ``opcode`` has no interval rule.
    """
    lookback = max(left.lookback, right.lookback)
    first_finite = max(left.first_finite, right.first_finite)
    can_finite = both_can_be_finite(left, right)
    const = const_pair(left, right)
    if opcode == int(Opcode.ADD):
        lo, hi = hull((left.lo + right.lo, left.hi + right.hi))
    elif opcode == int(Opcode.SUB):
        lo, hi = hull((left.lo - right.hi, left.hi - right.lo))
    elif opcode == int(Opcode.MUL):
        lo, hi = hull(
            tuple(_product(a, b) for a in (left.lo, left.hi) for b in (right.lo, right.hi))
        )
    elif opcode == int(Opcode.DIV):
        lo, hi = _quotient(left, right)
        # A divisor interval that holds zero also holds divisors that
        # fall back to ``fill``, so the result is no longer one value.
        const = const and not right.lo <= 0.0 <= right.hi
    else:
        raise ValueError(f"Opcode {opcode} has no interval certificate.")
    return Summary(lo, hi, lookback, first_finite, const, can_finite, "array")


def _quotient(left: Summary, right: Summary) -> tuple[float, float]:
    """Return the envelope of protected ``left / right``."""
    if right.lo <= 0.0 <= right.hi:
        # Divisors arbitrarily close to zero make the quotient unbounded.
        return _UNBOUNDED
    return hull(tuple(a / b for a in (left.lo, left.hi) for b in (right.lo, right.hi)))

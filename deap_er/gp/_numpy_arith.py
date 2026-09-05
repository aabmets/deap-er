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
from typing import Any

import numpy

__all__: list[str] = []

DEFAULT_FILL = 1.0
"""Value substituted for a non-finite result of a protected operation."""


def _protect(result: Any, fill: float, *inputs: Any) -> Any:
    """Replace non-finite results that were produced from finite inputs.

    A non-finite value that was already present in an input is left
    alone, so the ``nan`` warmup written by the window primitives
    survives a protected operation instead of being replaced by a
    fabricated number.

    Args:
        result: Raw result of the operation.
        fill: Value to substitute.
        *inputs: Operands the result was computed from.

    Returns:
        The result with fabricated non-finite values replaced.
    """
    finite = numpy.isfinite(result)
    if numpy.ndim(finite) == 0:
        if bool(finite):
            return result
        for item in inputs:
            if not bool(numpy.all(numpy.isfinite(item))):
                return result
        return numpy.float64(fill)

    bad = numpy.logical_not(finite)
    if not bool(bad.any()):
        return result

    okay = numpy.isfinite(inputs[0])
    for item in inputs[1:]:
        okay = numpy.logical_and(okay, numpy.isfinite(item))
    numpy.copyto(result, fill, where=numpy.logical_and(bad, okay))
    return result


def vadd(left: Any, right: Any) -> Any:
    """Add two series elementwise.

    Args:
        left: Left operand.
        right: Right operand.

    Returns:
        The elementwise sum.
    """
    return numpy.add(left, right)


def vsub(left: Any, right: Any) -> Any:
    """Subtract two series elementwise.

    Args:
        left: Left operand.
        right: Right operand.

    Returns:
        The elementwise difference.
    """
    return numpy.subtract(left, right)


def vmul(left: Any, right: Any) -> Any:
    """Multiply two series elementwise.

    Args:
        left: Left operand.
        right: Right operand.

    Returns:
        The elementwise product.
    """
    return numpy.multiply(left, right)


def vneg(value: Any) -> Any:
    """Negate a series elementwise.

    Args:
        value: Operand to negate.

    Returns:
        The elementwise negation.
    """
    return numpy.negative(value)


def vabs(value: Any) -> Any:
    """Take the elementwise absolute value of a series.

    Args:
        value: Operand.

    Returns:
        The elementwise absolute value.
    """
    return numpy.absolute(value)


def vdiv(left: Any, right: Any, fill: float = DEFAULT_FILL) -> Any:
    """Divide two series elementwise, protecting against zero divisors.

    Positions where finite operands produced a non-finite quotient are
    replaced by ``fill``. A non-finite value that came from an operand
    is preserved.

    Args:
        left: Numerator.
        right: Denominator.
        fill: Value substituted for a fabricated non-finite result.

    Returns:
        The protected elementwise quotient.
    """
    with numpy.errstate(divide="ignore", invalid="ignore"):
        result = numpy.divide(left, right)
    return _protect(result, fill, left, right)


def vlog(value: Any, fill: float = DEFAULT_FILL) -> Any:
    """Take the natural logarithm of a series, protecting the domain.

    Args:
        value: Operand.
        fill: Value substituted for a fabricated non-finite result.

    Returns:
        The protected elementwise logarithm.
    """
    with numpy.errstate(divide="ignore", invalid="ignore"):
        result = numpy.log(value)
    return _protect(result, fill, value)


def vsqrt(value: Any, fill: float = DEFAULT_FILL) -> Any:
    """Take the square root of a series, protecting the domain.

    Args:
        value: Operand.
        fill: Value substituted for a fabricated non-finite result.

    Returns:
        The protected elementwise square root.
    """
    with numpy.errstate(invalid="ignore"):
        result = numpy.sqrt(value)
    return _protect(result, fill, value)


def vsin(value: Any) -> Any:
    """Take the elementwise sine of a series.

    Args:
        value: Operand in radians.

    Returns:
        The elementwise sine.
    """
    return numpy.sin(value)


def vcos(value: Any) -> Any:
    """Take the elementwise cosine of a series.

    Args:
        value: Operand in radians.

    Returns:
        The elementwise cosine.
    """
    return numpy.cos(value)

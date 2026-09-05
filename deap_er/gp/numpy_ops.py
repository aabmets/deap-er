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
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy

from .columnar import Array, Mask, _reject_shadowed
from .primitives import PrimitiveSetTyped

__all__ = [
    "vadd",
    "vsub",
    "vmul",
    "vneg",
    "vabs",
    "vdiv",
    "vlog",
    "vsqrt",
    "vsin",
    "vcos",
    "vgt",
    "vlt",
    "vge",
    "vle",
    "veq",
    "vand",
    "vor",
    "vnot",
    "vwhere",
    "add_numpy_primitives",
    "infer_fill",
]

DEFAULT_FILL = 1.0
"""Value substituted for a non-finite result of a protected operation."""

_PROTECTED_NAMES = ("vdiv", "vlog", "vsqrt")


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


def vgt(left: Any, right: Any) -> Any:
    """Compare two series elementwise with ``>``.

    Args:
        left: Left operand.
        right: Right operand.

    Returns:
        A boolean mask.
    """
    return numpy.greater(left, right)


def vlt(left: Any, right: Any) -> Any:
    """Compare two series elementwise with ``<``.

    Args:
        left: Left operand.
        right: Right operand.

    Returns:
        A boolean mask.
    """
    return numpy.less(left, right)


def vge(left: Any, right: Any) -> Any:
    """Compare two series elementwise with ``>=``.

    Args:
        left: Left operand.
        right: Right operand.

    Returns:
        A boolean mask.
    """
    return numpy.greater_equal(left, right)


def vle(left: Any, right: Any) -> Any:
    """Compare two series elementwise with ``<=``.

    Args:
        left: Left operand.
        right: Right operand.

    Returns:
        A boolean mask.
    """
    return numpy.less_equal(left, right)


def veq(left: Any, right: Any) -> Any:
    """Compare two series elementwise with ``==``.

    Args:
        left: Left operand.
        right: Right operand.

    Returns:
        A boolean mask.
    """
    return numpy.equal(left, right)


def vand(left: Any, right: Any) -> Any:
    """Combine two masks elementwise with logical and.

    Args:
        left: Left mask.
        right: Right mask.

    Returns:
        A boolean mask.
    """
    return numpy.logical_and(left, right)


def vor(left: Any, right: Any) -> Any:
    """Combine two masks elementwise with logical or.

    Args:
        left: Left mask.
        right: Right mask.

    Returns:
        A boolean mask.
    """
    return numpy.logical_or(left, right)


def vnot(value: Any) -> Any:
    """Invert a mask elementwise.

    Args:
        value: Mask to invert.

    Returns:
        A boolean mask.
    """
    return numpy.logical_not(value)


def vwhere(condition: Any, on_true: Any, on_false: Any) -> Any:
    """Select elementwise between two series.

    This is how a program turns a condition into a value without a
    Python ``if``.

    Args:
        condition: Mask that selects the branch.
        on_true: Values taken where the mask is true.
        on_false: Values taken where the mask is false.

    Returns:
        The selected series.
    """
    return numpy.where(condition, on_true, on_false)


def infer_fill(prim_set: PrimitiveSetTyped) -> float:
    """Recover the protection fill that a primitive set was built with.

    Reads the value bound into the protected primitives by
    ``add_numpy_primitives``, so an alternative backend reproduces the
    same protection without being told the value again.

    Args:
        prim_set: Primitive set to inspect.

    Returns:
        The bound fill, or ``DEFAULT_FILL`` when the set carries no
        protected primitive.
    """
    for name in _PROTECTED_NAMES:
        func = prim_set.context.get(name)
        if isinstance(func, partial):
            value = func.keywords.get("fill")
            if value is not None:
                return float(value)
    return DEFAULT_FILL


def add_numpy_primitives(prim_set: PrimitiveSetTyped, *, fill: float = DEFAULT_FILL) -> None:
    """Register the vectorized primitive kit on a typed primitive set.

    Adds arithmetic, protected division and domain-limited unary
    operations, comparisons that return ``Mask``, mask logic, and
    ``vwhere``. The constants ``True`` and ``False`` are registered as
    ``Mask`` terminals so tree generation can terminate a mask branch.

    The chosen ``fill`` is bound into the protected primitives, where
    ``infer_fill`` can read it back so the other backends reproduce the
    same protection.

    Args:
        prim_set: Typed primitive set built by ``make_column_pset`` or
            an equivalent set over ``Array`` and ``Mask``.
        fill: Value substituted for a non-finite result of a protected
            operation that was computed from finite operands.

    Raises:
        ValueError: If a primitive name collides with an argument of
            ``prim_set``, or if a name is already registered.
    """
    binary: list[type] = [Array, Array]
    unary: list[type] = [Array]

    plain: dict[str, tuple[Callable[..., Any], list[type]]] = {
        "vadd": (vadd, binary),
        "vsub": (vsub, binary),
        "vmul": (vmul, binary),
        "vneg": (vneg, unary),
        "vabs": (vabs, unary),
        "vsin": (vsin, unary),
        "vcos": (vcos, unary),
    }
    protected: dict[str, tuple[Callable[..., Any], list[type]]] = {
        "vdiv": (vdiv, binary),
        "vlog": (vlog, unary),
        "vsqrt": (vsqrt, unary),
    }
    compares: dict[str, Callable[..., Any]] = {
        "vgt": vgt,
        "vlt": vlt,
        "vge": vge,
        "vle": vle,
        "veq": veq,
    }
    logic: dict[str, tuple[Callable[..., Any], list[type]]] = {
        "vand": (vand, [Mask, Mask]),
        "vor": (vor, [Mask, Mask]),
        "vnot": (vnot, [Mask]),
    }

    names = [*plain, *protected, *compares, *logic, "vwhere"]
    _reject_shadowed(prim_set, names)

    for name, (func, in_types) in plain.items():
        prim_set.add_primitive(func, in_types, Array, name)
    for name, (func, in_types) in protected.items():
        prim_set.add_primitive(partial(func, fill=fill), in_types, Array, name)
    for name, func in compares.items():
        prim_set.add_primitive(func, binary, Mask, name)
    for name, (func, in_types) in logic.items():
        prim_set.add_primitive(func, in_types, Mask, name)

    prim_set.add_primitive(vwhere, [Mask, Array, Array], Array, "vwhere")
    prim_set.add_terminal(True, Mask)
    prim_set.add_terminal(False, Mask)

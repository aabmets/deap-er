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

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
from typing import Any

import numpy
from numpy.lib.stride_tricks import sliding_window_view

from .columnar import Array, Window, reject_shadowed
from .primitives.primitive_set_typed import PrimitiveSetTyped
from .window_roll import as_series

__all__: list[str] = [
    "as_pair_series",
    "pair_moments",
    "rolling_corr",
    "rolling_cov",
    "rolling_beta",
    "add_pair_window_primitives",
]

type PairMoments = tuple[
    numpy.ndarray, int, numpy.ndarray | None, numpy.ndarray | None, numpy.ndarray | None
]


def as_pair_series(left: Any, right: Any, window: Any) -> tuple[numpy.ndarray, numpy.ndarray, int]:
    """Normalize two operands and a shared window length.

    Args:
        left: First series operand.
        right: Second series operand.
        window: Trailing window length. At least 1.

    Returns:
        Both operands as one-dimensional ``float64`` arrays and the
        window length as an ``int``.

    Raises:
        ValueError: If the window length is less than 1, or if an
            operand is not one-dimensional, or if the operands do not
            have the same length.
    """
    left_series, length = as_series(left, window)
    right_series, _ = as_series(right, window)
    if left_series.ndim != 1 or right_series.ndim != 1 or left_series.shape != right_series.shape:
        raise ValueError(
            f"Pair window operands must have the same length, got "
            f"shapes {left_series.shape} and {right_series.shape}."
        )
    return left_series, right_series, length


def pair_moments(left: Any, right: Any, window: Any) -> PairMoments:
    """Take population covariance and variances of a trailing pair window.

    The divisor is the window length. A negative variance from rounding
    is clamped to zero, matching ``rolling_std``.

    Args:
        left: First series.
        right: Second series.
        window: Trailing window length. At least 1.

    Returns:
        The ``nan``-prefilled output, the window length, and the
        covariance and variances of each full window, or None when
        the series is shorter than the window.

    Raises:
        ValueError: If the window length is less than 1, or if the
            operands are not one-dimensional series of the same length.
    """
    left_series, right_series, length = as_pair_series(left, right, window)
    result = numpy.full(left_series.shape, numpy.nan, dtype=numpy.float64)
    if length > left_series.size:
        return result, length, None, None, None
    left_view = sliding_window_view(left_series, length)
    right_view = sliding_window_view(right_series, length)
    count = float(length)
    mean_x = numpy.add.reduce(left_view, axis=-1) / count
    mean_y = numpy.add.reduce(right_view, axis=-1) / count
    squares_x = numpy.add.reduce(left_view * left_view, axis=-1) / count
    squares_y = numpy.add.reduce(right_view * right_view, axis=-1) / count
    products = numpy.add.reduce(left_view * right_view, axis=-1) / count
    cov = products - mean_x * mean_y
    var_x = numpy.maximum(squares_x - mean_x * mean_x, 0.0)
    var_y = numpy.maximum(squares_y - mean_y * mean_y, 0.0)
    return result, length, cov, var_x, var_y


def rolling_cov(left: Any, right: Any, window: Any) -> numpy.ndarray:
    """Take the population covariance of two series over a trailing window.

    The window is ``[t - window + 1, t]`` inclusive and the divisor is
    the window length. The first ``window - 1`` samples are ``nan``.

    Args:
        left: First series.
        right: Second series.
        window: Trailing window length. At least 1.

    Returns:
        The rolling covariance.

    Raises:
        ValueError: If the window length is less than 1, or if the
            operands are not one-dimensional series of the same length.
    """
    result, length, cov, _, _ = pair_moments(left, right, window)
    if cov is not None:
        result[length - 1 :] = cov
    return result


def rolling_corr(left: Any, right: Any, window: Any) -> numpy.ndarray:
    """Take the population correlation of two series over a trailing window.

    The window is ``[t - window + 1, t]`` inclusive. A window where
    either series has zero variance is ``nan``. The first
    ``window - 1`` samples are ``nan``.

    Args:
        left: First series.
        right: Second series.
        window: Trailing window length. At least 1.

    Returns:
        The rolling correlation.

    Raises:
        ValueError: If the window length is less than 1, or if the
            operands are not one-dimensional series of the same length.
    """
    result, length, cov, var_x, var_y = pair_moments(left, right, window)
    if cov is None or var_x is None or var_y is None:
        return result
    denom = numpy.sqrt(var_x * var_y)
    with numpy.errstate(invalid="ignore", divide="ignore"):
        values = cov / denom
    values[denom <= 0.0] = numpy.nan
    result[length - 1 :] = values
    return result


def rolling_beta(left: Any, right: Any, window: Any) -> numpy.ndarray:
    """Take the OLS slope of ``left`` on ``right`` over a trailing window.

    The slope is the population covariance divided by the population
    variance of ``right``. A window where ``right`` is constant is
    ``nan``. The first ``window - 1`` samples are ``nan``.

    Args:
        left: Dependent series.
        right: Independent series.
        window: Trailing window length. At least 1.

    Returns:
        The rolling beta of ``left`` versus ``right``.

    Raises:
        ValueError: If the window length is less than 1, or if the
            operands are not one-dimensional series of the same length.
    """
    result, length, cov, _, var_y = pair_moments(left, right, window)
    if cov is None or var_y is None:
        return result
    with numpy.errstate(invalid="ignore", divide="ignore"):
        values = cov / var_y
    values[var_y <= 0.0] = numpy.nan
    result[length - 1 :] = values
    return result


def add_pair_window_primitives(prim_set: PrimitiveSetTyped) -> None:
    """Register the two-input causal window kit on a typed primitive set.

    Every primitive takes two ``Array`` arguments and a ``Window`` and
    returns an ``Array``. All of them are causal: an output sample is a
    function of that sample and earlier ones only, and samples without
    enough history are ``nan``.

    Args:
        prim_set: Typed primitive set built by ``make_column_pset`` or
            an equivalent set over ``Array`` and ``Window``.

    Raises:
        ValueError: If a primitive name collides with an argument of
            ``prim_set``, or if a name is already registered.
    """
    windowed: dict[str, Callable[..., Any]] = {
        "rolling_corr": rolling_corr,
        "rolling_cov": rolling_cov,
        "rolling_beta": rolling_beta,
    }
    reject_shadowed(prim_set, list(windowed))

    in_types: list[type] = [Array, Array, Window]
    for name, func in windowed.items():
        prim_set.add_primitive(func, in_types, Array, name)

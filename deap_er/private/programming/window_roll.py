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
from numpy.lib.stride_tricks import sliding_window_view

__all__: list[str] = [
    "as_series",
    "delay",
    "diff",
    "rolling",
    "rolling_sum",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
]


def as_series(value: Any, window: Any) -> tuple[numpy.ndarray, int]:
    """Normalize an operand and a window length.

    Args:
        value: Series operand.
        window: Window length.

    Returns:
        The operand as a one-dimensional ``float64`` array, and the
        window length as an ``int``.

    Raises:
        ValueError: If the window length is less than 1.
    """
    length = int(window)
    if length < 1:
        raise ValueError(f"Window length must be at least 1, got {length}.")
    series = numpy.atleast_1d(numpy.asarray(value, dtype=numpy.float64))
    return series, length


def delay(value: Any, window: Any) -> numpy.ndarray:
    """Shift a series into the past by ``window`` samples.

    ``y[t]`` is ``x[t - window]``. The first ``window`` samples have no
    past to read and are ``nan``.

    Args:
        value: Series to shift.
        window: Number of samples to shift by. At least 1.

    Returns:
        The shifted series.

    Raises:
        ValueError: If the window length is less than 1.
    """
    series, length = as_series(value, window)
    result = numpy.full(series.shape, numpy.nan, dtype=numpy.float64)
    if length < series.size:
        result[length:] = series[:-length]
    return result


def diff(value: Any, window: Any) -> numpy.ndarray:
    """Subtract a delayed copy of a series from itself.

    ``y[t]`` is ``x[t] - x[t - window]``. The first ``window`` samples
    are ``nan``.

    Args:
        value: Series to difference.
        window: Number of samples to look back. At least 1.

    Returns:
        The differenced series.

    Raises:
        ValueError: If the window length is less than 1.
    """
    series, length = as_series(value, window)
    return series - delay(series, length)


def rolling(
    value: Any, window: Any, reducer: numpy.ufunc
) -> tuple[numpy.ndarray, int, numpy.ndarray | None]:
    """Reduce a trailing window over a series.

    The reduction runs over a strided view without materializing it,
    so peak memory stays proportional to the number of samples rather
    than to samples times window length.

    Args:
        value: Series to reduce.
        window: Trailing window length. At least 1.
        reducer: NumPy ufunc applied along the window axis.

    Returns:
        The ``nan``-prefilled output, the window length, and the
        reduction over each full window, or None when the series is
        shorter than the window.

    Raises:
        ValueError: If the window length is less than 1.
    """
    series, length = as_series(value, window)
    result = numpy.full(series.shape, numpy.nan, dtype=numpy.float64)
    if length > series.size:
        return result, length, None
    reduced = reducer.reduce(sliding_window_view(series, length), axis=-1)
    return result, length, numpy.asarray(reduced, dtype=numpy.float64)


def rolling_sum(value: Any, window: Any) -> numpy.ndarray:
    """Sum a trailing window of a series.

    The window is ``[t - window + 1, t]`` inclusive. The first
    ``window - 1`` samples are ``nan``.

    Args:
        value: Series to reduce.
        window: Trailing window length. At least 1.

    Returns:
        The rolling sum.

    Raises:
        ValueError: If the window length is less than 1.
    """
    result, length, reduced = rolling(value, window, numpy.add)
    if reduced is not None:
        result[length - 1 :] = reduced
    return result


def rolling_mean(value: Any, window: Any) -> numpy.ndarray:
    """Average a trailing window of a series.

    The window is ``[t - window + 1, t]`` inclusive. The first
    ``window - 1`` samples are ``nan``.

    Args:
        value: Series to reduce.
        window: Trailing window length. At least 1.

    Returns:
        The rolling mean.

    Raises:
        ValueError: If the window length is less than 1.
    """
    result, length, reduced = rolling(value, window, numpy.add)
    if reduced is not None:
        result[length - 1 :] = reduced / length
    return result


def rolling_std(value: Any, window: Any) -> numpy.ndarray:
    """Take the population standard deviation of a trailing window.

    The window is ``[t - window + 1, t]`` inclusive and the divisor is
    the window length. The first ``window - 1`` samples are ``nan``.

    Args:
        value: Series to reduce.
        window: Trailing window length. At least 1.

    Returns:
        The rolling standard deviation.

    Raises:
        ValueError: If the window length is less than 1.
    """
    series, length = as_series(value, window)
    result = numpy.full(series.shape, numpy.nan, dtype=numpy.float64)
    if length > series.size:
        return result
    mean = numpy.add.reduce(sliding_window_view(series, length), axis=-1) / length
    squares = numpy.add.reduce(sliding_window_view(series * series, length), axis=-1) / length
    variance = squares - mean * mean
    numpy.maximum(variance, 0.0, out=variance)
    result[length - 1 :] = numpy.sqrt(variance)
    return result


def rolling_min(value: Any, window: Any) -> numpy.ndarray:
    """Take the minimum of a trailing window of a series.

    The window is ``[t - window + 1, t]`` inclusive. A window that
    holds a ``nan`` reduces to ``nan``. The first ``window - 1``
    samples are ``nan``.

    Args:
        value: Series to reduce.
        window: Trailing window length. At least 1.

    Returns:
        The rolling minimum.

    Raises:
        ValueError: If the window length is less than 1.
    """
    result, length, reduced = rolling(value, window, numpy.minimum)
    if reduced is not None:
        result[length - 1 :] = reduced
    return result


def rolling_max(value: Any, window: Any) -> numpy.ndarray:
    """Take the maximum of a trailing window of a series.

    The window is ``[t - window + 1, t]`` inclusive. A window that
    holds a ``nan`` reduces to ``nan``. The first ``window - 1``
    samples are ``nan``.

    Args:
        value: Series to reduce.
        window: Trailing window length. At least 1.

    Returns:
        The rolling maximum.

    Raises:
        ValueError: If the window length is less than 1.
    """
    result, length, reduced = rolling(value, window, numpy.maximum)
    if reduced is not None:
        result[length - 1 :] = reduced
    return result

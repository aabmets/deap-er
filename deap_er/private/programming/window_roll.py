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
    "CHUNK_ROWS",
    "as_series",
    "window_deviations",
    "rolling",
    "rolling_sum",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
]

CHUNK_ROWS = 1024


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


def window_deviations(block: numpy.ndarray, length: int) -> numpy.ndarray:
    """Center each window of a strided block on its own mean.

    Subtracting the window mean before squaring is what keeps the
    moments accurate: ``E[x^2] - mean^2`` would cancel away the digits
    the variance is made of whenever the samples sit far from zero.

    A window holding an infinity has no finite mean, so its deviations
    are ``nan`` by definition rather than by accident.

    Args:
        block: Strided view of shape ``(rows, length)``.
        length: Window length.

    Returns:
        The deviations from each window's mean.
    """
    with numpy.errstate(invalid="ignore"):
        mean = numpy.add.reduce(block, axis=-1) / length
        return block - mean[:, None]


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

    Each window is centered on its own mean before squaring, so the
    result stays accurate however far the samples sit from zero. The
    windows are centered in row blocks, which keeps peak memory
    proportional to the block rather than to the whole series. A window
    that holds an infinity is ``nan``.

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
    view = sliding_window_view(series, length)
    output = result[length - 1 :]
    for start in range(0, view.shape[0], CHUNK_ROWS):
        block = view[start : start + CHUNK_ROWS]
        deviations = window_deviations(block, length)
        with numpy.errstate(invalid="ignore"):
            residue = numpy.add.reduce(deviations, axis=-1) / length
            squares = numpy.add.reduce(deviations * deviations, axis=-1) / length
            variance = squares - residue * residue
        numpy.maximum(variance, 0.0, out=variance)
        output[start : start + CHUNK_ROWS] = numpy.sqrt(variance)
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

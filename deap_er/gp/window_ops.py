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

from deap_er.rng import rng

from .columnar import Array, Window, _reject_shadowed
from .primitives import PrimitiveSetTyped

__all__ = [
    "delay",
    "diff",
    "rolling_sum",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "ema",
    "add_window_primitives",
    "add_window_ephemeral",
]

_samplers: dict[str, tuple[int, int, Callable[[], int]]] = {}


def _as_series(value: Any, window: Any) -> tuple[numpy.ndarray, int]:
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
    series, length = _as_series(value, window)
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
    series, length = _as_series(value, window)
    return series - delay(series, length)


def _rolling(
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
    series, length = _as_series(value, window)
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
    result, length, reduced = _rolling(value, window, numpy.add)
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
    result, length, reduced = _rolling(value, window, numpy.add)
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
    series, length = _as_series(value, window)
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
    result, length, reduced = _rolling(value, window, numpy.minimum)
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
    result, length, reduced = _rolling(value, window, numpy.maximum)
    if reduced is not None:
        result[length - 1 :] = reduced
    return result


def ema(value: Any, window: Any) -> numpy.ndarray:
    """Take a causal exponential moving average of a series.

    The recurrence is ``y[t] = a * x[t] + (1 - a) * y[t - 1]`` with
    ``a = 2 / (window + 1)``, seeded so that ``y`` equals ``x`` at the
    first finite sample. The first ``window - 1`` samples after that
    seed are reported as ``nan`` to match the rolling primitives. A
    ``nan`` inside the series propagates to every later sample.

    Args:
        value: Series to filter.
        window: Span of the average. At least 1.

    Returns:
        The exponential moving average.

    Raises:
        ValueError: If the window length is less than 1.
    """
    # Deferred so that importing deap_er does not pull in scipy.signal,
    # which no other part of the package needs.
    from scipy.signal import lfilter, lfilter_zi

    series, length = _as_series(value, window)
    result = numpy.full(series.shape, numpy.nan, dtype=numpy.float64)
    if length > series.size:
        return result

    finite = numpy.flatnonzero(numpy.isfinite(series))
    if finite.size == 0:
        return result

    start = int(finite[0])
    tail = series[start:]
    alpha = 2.0 / (length + 1.0)
    numer = numpy.array([alpha], dtype=numpy.float64)
    denom = numpy.array([1.0, alpha - 1.0], dtype=numpy.float64)
    state = lfilter_zi(numer, denom) * tail[0]
    filtered, _ = lfilter(numer, denom, tail, zi=state)

    result[start:] = filtered
    result[: start + length - 1] = numpy.nan
    return result


def add_window_primitives(prim_set: PrimitiveSetTyped) -> None:
    """Register the causal window primitive kit on a typed primitive set.

    Every primitive takes an ``Array`` and a ``Window`` and returns an
    ``Array``. All of them are causal: an output sample is a function
    of that sample and earlier ones only, and samples without enough
    history are ``nan`` rather than zero, so a consumer can mask the
    warmup instead of trading on fabricated values.

    Args:
        prim_set: Typed primitive set built by ``make_column_pset`` or
            an equivalent set over ``Array`` and ``Window``.

    Raises:
        ValueError: If a primitive name collides with an argument of
            ``prim_set``, or if a name is already registered.
    """
    windowed: dict[str, Callable[..., Any]] = {
        "delay": delay,
        "diff": diff,
        "rolling_sum": rolling_sum,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "rolling_min": rolling_min,
        "rolling_max": rolling_max,
        "ema": ema,
    }
    _reject_shadowed(prim_set, list(windowed))

    in_types: list[type] = [Array, Window]
    for name, func in windowed.items():
        prim_set.add_primitive(func, in_types, Array, name)


def add_window_ephemeral(prim_set: PrimitiveSetTyped, name: str, low: int, high: int) -> None:
    """Register a random window length as an ephemeral constant.

    Each tree samples its own immutable window length from the closed
    interval ``[low, high]``. The sampler is memoized by name, because
    a primitive set rejects two ephemerals that share a name but not a
    function.

    Args:
        prim_set: Typed primitive set to register on.
        name: Name of this ephemeral type. Must be unique across every
            primitive set in the process.
        low: Smallest window length that may be sampled. At least 1.
        high: Largest window length that may be sampled.

    Raises:
        ValueError: If the bounds are invalid, or if ``name`` was
            already used with different bounds.
    """
    if low < 1:
        raise ValueError(f"The lowest window length must be at least 1, got {low}.")
    if high < low:
        raise ValueError(f"Window bounds are inverted: [{low}, {high}].")

    known = _samplers.get(name)
    if known is None:

        def sampler() -> int:
            return rng.randint(low, high)

        _samplers[name] = (low, high, sampler)
    elif known[:2] != (low, high):
        raise ValueError(
            f"The window ephemeral '{name}' was already registered with "
            f"bounds [{known[0]}, {known[1]}]. Use a different name."
        )

    prim_set.add_ephemeral_constant(name, _samplers[name][2], Window)

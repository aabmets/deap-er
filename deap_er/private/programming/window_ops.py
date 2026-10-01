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
from typing import Any, override

import numpy

from deap_er.private.various.rng import rng

from .columnar import Array, Window, reject_shadowed
from .primitives.primitive_set_typed import PrimitiveSetTyped
from .window_roll import (
    as_series,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_std,
    rolling_sum,
)
from .window_shift import delay, diff

__all__: list[str] = ["ema", "add_window_primitives", "add_window_ephemeral"]


class _WindowSampler:
    """Draw a window length from ``[low, high]``.

    Pickles by name and bounds and unpickles to the process's memoized
    sampler, so a tree that holds a window ephemeral loads before the
    ephemeral is registered and still matches a later registration.
    """

    __slots__ = ("name", "low", "high")

    def __init__(self, name: str, low: int, high: int) -> None:
        """Store the sampler name and its inclusive bounds."""
        self.name = name
        self.low = low
        self.high = high

    def __call__(self) -> int:
        """Return a random window length."""
        return rng.randint(self.low, self.high)

    @override
    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle as a lookup of the memoized sampler."""
        return _window_sampler, (self.name, self.low, self.high)


_samplers: dict[str, _WindowSampler] = {}


def _window_sampler(name: str, low: int, high: int) -> _WindowSampler:
    """Return the sampler memoized under ``name``, creating it on first use.

    Raises:
        ValueError: If the bounds are invalid, or if ``name`` was
            already used with different bounds.
    """
    if low < 1:
        raise ValueError(f"The lowest window length must be at least 1, got {low}.")
    if high < low:
        raise ValueError(f"Window bounds are inverted: [{low}, {high}].")
    sampler = _samplers.setdefault(name, _WindowSampler(name, low, high))
    if (sampler.low, sampler.high) != (low, high):
        raise ValueError(
            f"The window ephemeral '{name}' was already registered with "
            f"bounds [{sampler.low}, {sampler.high}]. Use a different name."
        )
    return sampler


def ema(value: Any, window: Any) -> numpy.ndarray:
    """Take a causal exponential moving average of a series.

    The recurrence is ``y[t] = a * x[t] + (1 - a) * y[t - 1]`` with
    ``a = 2 / (window + 1)``. A non-finite sample is ``nan`` in the
    output and ends the current segment. Each run of finite samples is
    its own segment: it is seeded so that ``y`` equals ``x`` at its
    first sample, and its first ``window - 1`` samples are reported as
    ``nan`` to match the rolling primitives. A gap therefore never
    poisons the rest of the series, and a series packed after ``nan``
    padding gets the same average as that series alone.

    The average depends on every earlier sample of its segment, so it
    has no finite lookback. :func:`~deap_er.gp.tape_lookback` raises
    :class:`~deap_er.gp.UnboundedLookbackError` for it.

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

    series, length = as_series(value, window)
    result = numpy.full(series.shape, numpy.nan, dtype=numpy.float64)
    if length > series.size:
        return result

    alpha = 2.0 / (length + 1.0)
    numer = numpy.array([alpha], dtype=numpy.float64)
    denom = numpy.array([1.0, alpha - 1.0], dtype=numpy.float64)
    zi = lfilter_zi(numer, denom)

    finite = numpy.isfinite(series).astype(numpy.int8)
    edges = numpy.flatnonzero(numpy.diff(finite, prepend=0, append=0))
    for start, stop in zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True):
        if stop - start < length:
            continue
        segment = series[start:stop]
        filtered, _ = lfilter(numer, denom, segment, zi=zi * segment[0])
        result[start + length - 1 : stop] = filtered[length - 1 :]
    return result


_WINDOWED: dict[str, Callable[..., Any]] = {
    "delay": delay,
    "diff": diff,
    "rolling_sum": rolling_sum,
    "rolling_mean": rolling_mean,
    "rolling_std": rolling_std,
    "rolling_min": rolling_min,
    "rolling_max": rolling_max,
    "ema": ema,
}


def add_window_primitives(prim_set: PrimitiveSetTyped, *, ema: bool = True) -> None:
    """Register the causal window primitive kit on a typed primitive set.

    Every primitive takes an ``Array`` and a ``Window`` and returns an
    ``Array``. All of them are causal: an output sample is a function
    of that sample and earlier ones only, and samples without enough
    history are ``nan`` rather than zero, so a consumer can mask the
    warmup instead of trading on fabricated values.

    Args:
        prim_set: Typed primitive set built by ``make_column_pset`` or
            an equivalent set over ``Array`` and ``Window``.
        ema: If False, leave out ``ema``, whose output has no finite
            lookback. Defaults to True.

    Raises:
        ValueError: If a primitive name collides with an argument of
            ``prim_set``, or if a name is already registered.
    """
    windowed = {key: func for key, func in _WINDOWED.items() if ema or key != "ema"}
    reject_shadowed(prim_set, list(windowed))

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
        name: Name of this ephemeral type. The sampler is shared by
            name across the process, so one name can only ever carry
            one ``[low, high]`` pair. ``columnar_pset`` derives
            ``window_{low}_{high}`` when no name is given.
        low: Smallest window length that may be sampled. At least 1.
        high: Largest window length that may be sampled.

    Raises:
        ValueError: If the bounds are invalid, or if ``name`` was
            already used with different bounds.
    """
    prim_set.add_ephemeral_constant(name, _window_sampler(name, low, high), Window)

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

__all__: list[str] = ["ts_rank", "ts_argmax", "ts_argmin", "add_ts_primitives"]

type WindowView = tuple[numpy.ndarray, int, numpy.ndarray | None]


def _windows(value: Any, window: Any) -> WindowView:
    """Normalize a series and open a trailing-window view.

    Args:
        value: Series operand.
        window: Trailing window length. At least 1.

    Returns:
        The ``nan``-prefilled output, the window length, and the
        strided view, or None when the series is shorter than the
        window.

    Raises:
        ValueError: If the window length is less than 1.
    """
    series, length = as_series(value, window)
    result = numpy.full(series.shape, numpy.nan, dtype=numpy.float64)
    if length > series.size:
        return result, length, None
    return result, length, sliding_window_view(series, length)


def ts_rank(value: Any, window: Any) -> numpy.ndarray:
    """Take the percentile rank of the current sample in a trailing window.

    The window is ``[t - window + 1, t]`` inclusive. The value is the
    1-based average rank of ``x[t]`` among those samples, scaled by
    ``(rank - 1) / (window - 1)``. A unique window low is ``0.0``, a
    unique window high is ``1.0``, and an all-tie window is ``0.5``.
    A window of 1 is ``nan``. A window that holds a ``nan`` is
    ``nan``. The first ``window - 1`` samples are ``nan``.

    Args:
        value: Series to rank.
        window: Trailing window length. At least 1.

    Returns:
        The rolling percentile rank.

    Raises:
        ValueError: If the window length is less than 1.
    """
    result, length, view = _windows(value, window)
    if view is None or length == 1:
        return result
    current = view[:, -1:]
    less = numpy.add.reduce(view < current, axis=-1)
    equal = numpy.add.reduce(view == current, axis=-1)
    rank = less + (equal + 1.0) / 2.0
    values = (rank - 1.0) / (length - 1.0)
    values[numpy.isnan(view).any(axis=-1)] = numpy.nan
    result[length - 1 :] = values
    return result


def _ts_arg(value: Any, window: Any, locator: Callable[..., Any]) -> numpy.ndarray:
    """Take the age of a trailing-window extremum.

    Age ``0`` is the current sample. Ties keep the most recent
    extremum by locating it on the reversed window.

    Args:
        value: Series to search.
        window: Trailing window length. At least 1.
        locator: ``argmax`` or ``argmin`` applied to the reversed
            window.

    Returns:
        The age of the extremum, or ``nan``.

    Raises:
        ValueError: If the window length is less than 1.
    """
    result, length, view = _windows(value, window)
    if view is None:
        return result
    ages = numpy.asarray(locator(view[:, ::-1], axis=-1), dtype=numpy.float64)
    ages[numpy.isnan(view).any(axis=-1)] = numpy.nan
    result[length - 1 :] = ages
    return result


def ts_argmax(value: Any, window: Any) -> numpy.ndarray:
    """Take how many samples ago the trailing-window maximum occurred.

    The window is ``[t - window + 1, t]`` inclusive. ``0`` means the
    current sample is the maximum. A tie keeps the most recent
    maximum. A window that holds a ``nan`` is ``nan``. The first
    ``window - 1`` samples are ``nan``.

    Args:
        value: Series to search.
        window: Trailing window length. At least 1.

    Returns:
        The age of the window maximum.

    Raises:
        ValueError: If the window length is less than 1.
    """
    return _ts_arg(value, window, numpy.argmax)


def ts_argmin(value: Any, window: Any) -> numpy.ndarray:
    """Take how many samples ago the trailing-window minimum occurred.

    The window is ``[t - window + 1, t]`` inclusive. ``0`` means the
    current sample is the minimum. A tie keeps the most recent
    minimum. A window that holds a ``nan`` is ``nan``. The first
    ``window - 1`` samples are ``nan``.

    Args:
        value: Series to search.
        window: Trailing window length. At least 1.

    Returns:
        The age of the window minimum.

    Raises:
        ValueError: If the window length is less than 1.
    """
    return _ts_arg(value, window, numpy.argmin)


def add_ts_primitives(prim_set: PrimitiveSetTyped) -> None:
    """Register the causal time-series unary kit on a typed primitive set.

    Every primitive takes an ``Array`` and a ``Window`` and returns an
    ``Array``. All of them are causal: an output sample is a function
    of that sample and earlier ones only, and samples without enough
    history are ``nan``.

    Args:
        prim_set: Typed primitive set built by ``make_column_pset`` or
            an equivalent set over ``Array`` and ``Window``.

    Raises:
        ValueError: If a primitive name collides with an argument of
            ``prim_set``, or if a name is already registered.
    """
    windowed: dict[str, Callable[..., Any]] = {
        "ts_rank": ts_rank,
        "ts_argmax": ts_argmax,
        "ts_argmin": ts_argmin,
    }
    reject_shadowed(prim_set, list(windowed))

    in_types: list[type] = [Array, Window]
    for name, func in windowed.items():
        prim_set.add_primitive(func, in_types, Array, name)

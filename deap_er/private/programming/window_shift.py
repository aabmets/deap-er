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

from .window_roll import as_series

__all__: list[str] = ["delay", "diff"]


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

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
from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy

__all__: list[str] = ["case_errors"]


def _as_series(
    predicted: numpy.ndarray,
    target: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray, int]:
    left = numpy.asarray(predicted, dtype=numpy.float64)
    right = numpy.asarray(target, dtype=numpy.float64)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("predicted and target must be one-dimensional arrays")
    if left.shape[0] != right.shape[0]:
        raise ValueError("predicted and target must have the same length")
    return left, right, int(left.shape[0])


def _validate_interval(start: object, stop: object, length: int) -> tuple[int, int]:
    if isinstance(start, bool) or isinstance(stop, bool):
        raise ValueError("range endpoints must be integers")
    if not isinstance(start, Integral) or not isinstance(stop, Integral):
        raise ValueError("range endpoints must be integers")
    begin = int(start)
    end = int(stop)
    if begin < 0 or end < 0 or begin > end or end > length:
        raise ValueError("range endpoints must satisfy 0 <= start <= stop <= length")
    return begin, end


def _ranges_from_mask(mask: numpy.ndarray, length: int) -> list[tuple[int, int]]:
    if mask.ndim != 1:
        raise ValueError("a boolean mask must be one-dimensional")
    if mask.shape[0] != length:
        raise ValueError("a boolean mask must match the series length")
    if mask.dtype != bool:
        raise ValueError("a boolean mask must have dtype bool")
    indices = numpy.flatnonzero(mask)
    if indices.size == 0:
        return []
    breaks = numpy.flatnonzero(numpy.diff(indices) > 1) + 1
    starts = numpy.split(indices, breaks)
    return [(int(run[0]), int(run[-1]) + 1) for run in starts]


def _ranges_from_array(ranges: numpy.ndarray, length: int) -> list[tuple[int, int]]:
    if ranges.dtype == bool:
        return _ranges_from_mask(ranges, length)
    if ranges.ndim == 2 and ranges.shape[1] == 2 and numpy.issubdtype(ranges.dtype, numpy.integer):
        return [_validate_interval(start, stop, length) for start, stop in ranges]
    if ranges.ndim == 1 and numpy.issubdtype(ranges.dtype, numpy.integer):
        raise ValueError(
            "integer arrays are not accepted as case boundaries; "
            "pass explicit (start, stop) pairs, a (n_cases, 2) integer array, "
            "or a one-dimensional boolean mask"
        )
    raise ValueError(
        "ranges must be explicit (start, stop) pairs, a (n_cases, 2) integer "
        "array, or a one-dimensional boolean mask"
    )


def _normalize_ranges(
    ranges: Sequence[tuple[int, int]] | numpy.ndarray,
    length: int,
) -> list[tuple[int, int]]:
    if isinstance(ranges, numpy.ndarray):
        return _ranges_from_array(ranges, length)
    return [_validate_interval(start, stop, length) for start, stop in ranges]


def _finite_mask(predicted: numpy.ndarray, target: numpy.ndarray) -> numpy.ndarray:
    return numpy.isfinite(predicted) & numpy.isfinite(target)


def _case_mse(
    predicted: numpy.ndarray,
    target: numpy.ndarray,
    valid: numpy.ndarray,
    start: int,
    stop: int,
    empty: float,
) -> float:
    sample = valid[start:stop]
    if not numpy.any(sample):
        return empty
    diff = predicted[start:stop][sample] - target[start:stop][sample]
    return float(numpy.mean(diff * diff))


def case_errors(
    predicted: numpy.ndarray,
    target: numpy.ndarray,
    ranges: Sequence[tuple[int, int]] | numpy.ndarray,
    *,
    valid: numpy.ndarray | None = None,
    empty: float = float("inf"),
) -> tuple[float, ...]:
    """Return one mean-squared error per case segment.

    Each case is a half-open interval ``[start, stop)`` over aligned
    ``predicted`` and ``target`` series. Only samples marked valid inside
    the interval are scored. By default a sample is valid when both series
    are finite at that index.

    Pass an explicit ``valid`` mask when comparisons or ``vwhere`` can
    hide a ``nan`` warmup while the prediction stays finite. That mask is
    intersected with the finite check, so non-finite samples never enter
    the MSE even when ``valid`` marks them ``True``. Segment boundaries
    stay on the caller; this helper does not split a series
    chronologically.

    Args:
        predicted: Predicted series.
        target: Target series, same length as ``predicted``.
        ranges: A sequence of ``(start, stop)`` pairs, a ``(n_cases, 2)``
            integer array of half-open bounds, or a one-dimensional
            ``bool`` mask. A mask defines one case per contiguous run of
            ``True`` values.
        valid: Optional per-sample mask intersected with the finite check.
            Use it to drop trusted prefixes such as hidden warmup.
        empty: Value returned when a case has no scorable samples.

    Returns:
        One MSE per case, in range order.

    Raises:
        ValueError: If the inputs are not aligned one-dimensional arrays,
            if range endpoints are invalid, or if a mask has the wrong
            shape or dtype.
    """
    predicted, target, length = _as_series(predicted, target)
    intervals = _normalize_ranges(ranges, length)
    finite = _finite_mask(predicted, target)
    if valid is None:
        sample_valid = finite
    else:
        sample_valid = numpy.asarray(valid, dtype=bool)
        if sample_valid.ndim != 1 or sample_valid.shape[0] != length:
            raise ValueError("valid must be a one-dimensional mask matching the series length")
        sample_valid = sample_valid & finite
    return tuple(
        _case_mse(predicted, target, sample_valid, start, stop, empty) for start, stop in intervals
    )

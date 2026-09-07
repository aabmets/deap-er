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

__all__: list[str] = [
    "mask_from_ranges",
    "normalize_case_ranges",
    "ranges_from_mask",
]


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


def ranges_from_mask(mask: numpy.ndarray, length: int) -> list[tuple[int, int]]:
    """Split a boolean mask into one half-open range per ``True`` run.

    Args:
        mask: One-dimensional ``bool`` mask.
        length: Expected mask length.

    Returns:
        Contiguous ``[start, stop)`` runs of ``True``.

    Raises:
        ValueError: If ``mask`` is not a 1-D ``bool`` array of ``length``.
    """
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
        return ranges_from_mask(ranges, length)
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


def normalize_case_ranges(
    ranges: Sequence[tuple[int, int]] | numpy.ndarray,
    length: int,
) -> list[tuple[int, int]]:
    """Normalize ranges or a mask into validated half-open intervals.

    Args:
        ranges: ``(start, stop)`` pairs, a ``(n_cases, 2)`` integer array,
            or a one-dimensional ``bool`` mask.
        length: Exclusive upper bound for endpoints, or the mask length.

    Returns:
        Validated ``[start, stop)`` intervals in caller order.

    Raises:
        ValueError: If a range or mask is invalid.
    """
    if isinstance(ranges, numpy.ndarray):
        return _ranges_from_array(ranges, length)
    return [_validate_interval(start, stop, length) for start, stop in ranges]


def mask_from_ranges(
    ranges: Sequence[tuple[int, int]] | numpy.ndarray,
    length: int,
) -> numpy.ndarray:
    """Paint validated ranges onto a boolean mask of ``length``.

    Args:
        ranges: The same range or mask input as ``normalize_case_ranges``.
        length: Length of the returned mask.

    Returns:
        A 1-D ``bool`` mask that is ``True`` on every painted interval.

    Raises:
        ValueError: If a range or mask is invalid.
    """
    mask = numpy.zeros(length, dtype=bool)
    for start, stop in normalize_case_ranges(ranges, length):
        mask[start:stop] = True
    return mask

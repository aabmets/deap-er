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

import math
import numbers
from collections.abc import Sequence
from typing import cast

__all__: list[str] = [
    "MAX_GRID_CELLS",
    "descriptor_to_index",
    "index_to_descriptor_center",
    "parse_grid_config",
]

MAX_GRID_CELLS = 10**9


def parse_grid_config(
    ranges: Sequence[tuple[float, float]],
    bins: Sequence[int] | int,
) -> tuple[tuple[tuple[float, float], ...], tuple[int, ...], int]:
    """Validate grid bounds and resolution.

    Args:
        ranges: ``(low, high)`` bounds per behavior dimension.
        bins: Resolution per dimension, or one integer for every
            dimension.

    Returns:
        Normalized ranges, bins, and total cell count.

    Raises:
        ValueError: If the configuration is invalid.
    """
    if not ranges:
        raise ValueError("ranges must contain at least one dimension")
    ranges_tuple = tuple((float(low), float(high)) for low, high in ranges)
    for low, high in ranges_tuple:
        if low >= high:
            raise ValueError("each range must satisfy low < high")
    if isinstance(bins, numbers.Integral) and not isinstance(bins, bool):
        bins_value = int(bins)
        if bins_value < 1:
            raise ValueError("bins must be at least 1")
        bins_tuple = tuple(bins_value for _ in ranges_tuple)
    else:
        if not isinstance(bins, Sequence) or isinstance(bins, (str, bytes)):
            raise ValueError("bins must be an integer or a sequence of integers")
        bins_seq = cast(Sequence[int], bins)
        bins_tuple = tuple(int(value) for value in bins_seq)
        if len(bins_tuple) != len(ranges_tuple):
            raise ValueError("bins must match the number of ranges")
        if any(value < 1 for value in bins_tuple):
            raise ValueError("each bin count must be at least 1")
    num_cells = math.prod(bins_tuple)
    if num_cells > MAX_GRID_CELLS:
        raise ValueError(f"grid has {num_cells} cells, which exceeds {MAX_GRID_CELLS}")
    return ranges_tuple, bins_tuple, num_cells


def descriptor_to_index(
    descriptor: Sequence[float],
    ranges: Sequence[tuple[float, float]],
    bins: Sequence[int],
) -> tuple[int, ...]:
    """Map a behavior descriptor to its grid cell.

    Coordinates outside ``ranges`` are clipped before binning.
    """
    if len(descriptor) != len(ranges):
        raise ValueError(
            f"descriptor length {len(descriptor)} does not match {len(ranges)} dimensions"
        )
    index: list[int] = []
    for value, (low, high), cell_bins in zip(descriptor, ranges, bins, strict=True):
        clipped = min(max(float(value), low), high)
        span = high - low
        cell = int((clipped - low) / span * cell_bins)
        if cell >= cell_bins:
            cell = cell_bins - 1
        index.append(cell)
    return tuple(index)


def index_to_descriptor_center(
    index: tuple[int, ...],
    ranges: Sequence[tuple[float, float]],
    bins: Sequence[int],
) -> tuple[float, ...]:
    """Return the center of a grid cell in behavior space."""
    if len(index) != len(ranges):
        raise ValueError(f"index length {len(index)} does not match {len(ranges)} dimensions")
    centers: list[float] = []
    for cell, (low, high), cell_bins in zip(index, ranges, bins, strict=True):
        if cell < 0 or cell >= cell_bins:
            raise ValueError(f"index coordinate {cell} is out of range for {cell_bins} bins")
        span = high - low
        centers.append(low + (cell + 0.5) * span / cell_bins)
    return tuple(centers)

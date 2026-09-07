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

import numpy

from deap_er.private.various.rng import rng

__all__: list[str] = ["mut_case_mask", "mut_case_ranges"]


def mut_case_ranges(
    ranges: list[tuple[int, int]],
    *,
    length: int,
    mut_prob: float,
) -> tuple[list[tuple[int, int]]]:
    """Jitter half-open case ranges in place.

    Each interval is mutated independently with probability ``mut_prob``.
    Endpoints stay inside ``[0, length]`` and are swapped if they cross.
    ``mut_prob <= 0`` is a no-op.

    Args:
        ranges: Mutable list of ``(start, stop)`` pairs.
        length: Exclusive upper bound for ``stop``.
        mut_prob: Probability of mutating each interval.

    Returns:
        A one-element tuple containing ``ranges``.

    Raises:
        ValueError: If ``length`` is negative.
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    if mut_prob <= 0 or not ranges:
        return (ranges,)
    draws = rng.take_floats(len(ranges))
    for i, (start, stop) in enumerate(ranges):
        if draws[i] < mut_prob:
            ranges[i] = _jitter_interval(int(start), int(stop), length)
    return (ranges,)


def mut_case_mask(mask: numpy.ndarray, *, mut_prob: float) -> tuple[numpy.ndarray]:
    """Flip contiguous True/False runs of a boolean mask in place.

    Each run is flipped independently with probability ``mut_prob``.
    ``mut_prob <= 0`` is a no-op.

    Args:
        mask: One-dimensional mutable boolean array.
        mut_prob: Probability of flipping each contiguous run.

    Returns:
        A one-element tuple containing ``mask``.
    """
    if mut_prob <= 0:
        return (mask,)
    runs = _mask_runs(mask)
    if not runs:
        return (mask,)
    draws = rng.take_floats(len(runs))
    for (start, stop), draw in zip(runs, draws, strict=True):
        if draw < mut_prob:
            slc = mask[start:stop]
            slc[:] = ~slc
    return (mask,)


def _jitter_interval(start: int, stop: int, length: int) -> tuple[int, int]:
    span = max(stop - start, 1)
    delta = max(1, span // 4)
    start += rng.randint(-delta, delta)
    stop += rng.randint(-delta, delta)
    start = min(max(start, 0), length)
    stop = min(max(stop, 0), length)
    if start > stop:
        start, stop = stop, start
    return start, stop


def _mask_runs(mask: numpy.ndarray) -> list[tuple[int, int]]:
    size = int(mask.shape[0])
    if size == 0:
        return []
    runs: list[tuple[int, int]] = []
    begin = 0
    current = bool(mask[0])
    for i in range(1, size):
        value = bool(mask[i])
        if value != current:
            runs.append((begin, i))
            begin = i
            current = value
    runs.append((begin, size))
    return runs

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

__all__: list[str] = [
    "block_sum",
    "pairwise_sum",
    "scan_sum",
    "scan_variance",
    "scan_pair_moments",
]


def block_sum(stack: Any, row: int, begin: int, count: int) -> float:  # pragma: no cover
    """Sum up to 128 samples in the order NumPy's pairwise ``add`` uses.

    Fewer than 8 samples are added in turn; more go to eight
    interleaved partial sums, and the remainder is added in turn.

    Args:
        stack: Column-length workspace.
        row: Stack row that holds the series.
        begin: Inclusive start index.
        count: Number of samples, at most 128.

    Returns:
        The sum of the samples.
    """
    if count < 8:
        total = 0.0
        for index in range(begin, begin + count):
            total += float(stack[row, index])
        return total
    r0 = float(stack[row, begin])
    r1 = float(stack[row, begin + 1])
    r2 = float(stack[row, begin + 2])
    r3 = float(stack[row, begin + 3])
    r4 = float(stack[row, begin + 4])
    r5 = float(stack[row, begin + 5])
    r6 = float(stack[row, begin + 6])
    r7 = float(stack[row, begin + 7])
    step = 8
    while step < count - count % 8:
        at = begin + step
        r0 += float(stack[row, at])
        r1 += float(stack[row, at + 1])
        r2 += float(stack[row, at + 2])
        r3 += float(stack[row, at + 3])
        r4 += float(stack[row, at + 4])
        r5 += float(stack[row, at + 5])
        r6 += float(stack[row, at + 6])
        r7 += float(stack[row, at + 7])
        step += 8
    total = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    for index in range(begin + step, begin + count):
        total += float(stack[row, index])
    return total


def pairwise_sum(stack: Any, row: int, begin: int, count: int) -> float:  # pragma: no cover
    """Sum more than 128 samples in the order NumPy's pairwise ``add`` uses.

    The run splits in half on a multiple of 8 until each part fits
    ``block_sum``.

    Args:
        stack: Column-length workspace.
        row: Stack row that holds the series.
        begin: Inclusive start index.
        count: Number of samples.

    Returns:
        The sum of the samples.
    """
    half = count // 2
    half -= half % 8
    left = (
        block_sum(stack, row, begin, half) if half <= 128 else pairwise_sum(stack, row, begin, half)
    )
    rest = count - half
    right = (
        block_sum(stack, row, begin + half, rest)
        if rest <= 128
        else pairwise_sum(stack, row, begin + half, rest)
    )
    return left + right


def scan_sum(stack: Any, row: int, begin: int, end: int) -> float:  # pragma: no cover
    """Sum one window from its own samples.

    Nothing carries over from earlier windows, so a value depends only
    on the samples in its window. The samples are added in the order
    of ``add.reduce`` (see ``block_sum`` and ``pairwise_sum``), so the
    sum matches the opcode backend bit for bit, ``nan`` and infinities
    included.

    Args:
        stack: Column-length workspace.
        row: Stack row that holds the series.
        begin: Inclusive start index.
        end: Exclusive stop index.

    Returns:
        The sum of the window.
    """
    count = end - begin
    if count <= 128:
        return 0.0 + block_sum(stack, row, begin, count)
    return 0.0 + pairwise_sum(stack, row, begin, count)


def scan_variance(stack: Any, row: int, begin: int, end: int) -> float:  # pragma: no cover
    """Take one window's variance around its own mean.

    Centering on the window mean is what the Python backend does, so
    the two agree without either of them having to round alike.

    Args:
        stack: Column-length workspace.
        row: Stack row that holds the series.
        begin: Inclusive start index.
        end: Exclusive stop index.

    Returns:
        The population variance of the window.
    """
    count = float(end - begin)
    total = 0.0
    for index in range(begin, end):
        total += float(stack[row, index])
    mean = total / count
    residue = 0.0
    squares = 0.0
    for index in range(begin, end):
        deviation = float(stack[row, index]) - mean
        residue += deviation
        squares += deviation * deviation
    residue /= count
    return squares / count - residue * residue


def scan_pair_moments(  # pragma: no cover
    stack: Any, left_row: int, right_row: int, begin: int, end: int
) -> tuple[float, float, float]:
    """Take one pair window's moments around their own means.

    Args:
        stack: Column-length workspace.
        left_row: Left-series stack row.
        right_row: Right-series stack row.
        begin: Inclusive start index.
        end: Exclusive stop index.

    Returns:
        The covariance and the two variances of the window.
    """
    count = float(end - begin)
    mean_x = 0.0
    mean_y = 0.0
    for index in range(begin, end):
        mean_x += float(stack[left_row, index])
        mean_y += float(stack[right_row, index])
    mean_x /= count
    mean_y /= count
    residue_x = 0.0
    residue_y = 0.0
    squares_x = 0.0
    squares_y = 0.0
    products = 0.0
    for index in range(begin, end):
        dev_x = float(stack[left_row, index]) - mean_x
        dev_y = float(stack[right_row, index]) - mean_y
        residue_x += dev_x
        residue_y += dev_y
        squares_x += dev_x * dev_x
        squares_y += dev_y * dev_y
        products += dev_x * dev_y
    residue_x /= count
    residue_y /= count
    return (
        products / count - residue_x * residue_y,
        squares_x / count - residue_x * residue_x,
        squares_y / count - residue_y * residue_y,
    )

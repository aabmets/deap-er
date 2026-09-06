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
import math
from typing import Any

__all__: list[str] = [
    "EPSILON",
    "TRUST_MARGIN",
    "row_offset",
    "variance_untrusted",
    "scan_sums",
    "scan_variance",
    "scan_pair_sums",
    "scan_pair_moments",
]

EPSILON = 2.220446049250313e-16
TRUST_MARGIN = 1e-10


def row_offset(stack: Any, row: int, begin: int, end: int) -> float:  # pragma: no cover
    """Pick the shift that keeps a row's running moments conditioned.

    The running sums only have to survive the spread of a window, not
    its distance from zero, so they are kept relative to one sample.
    Reading the shift from the window being rebuilt, rather than from
    the start of the column, keeps a series that wanders away from its
    first sample on the running path.

    Args:
        stack: Column-length workspace.
        row: Stack row to inspect.
        begin: Inclusive start index.
        end: Exclusive stop index.

    Returns:
        The first finite sample in the range, or ``0.0`` when the range
        holds no finite sample.
    """
    for index in range(begin, end):
        value = stack[row, index]
        if math.isfinite(value):
            return float(value)
    return 0.0


def variance_untrusted(variance: float, err_bound: float, arg: int) -> int:  # pragma: no cover
    """Report whether a running variance has lost too many digits.

    Every sample that enters or leaves the window leaves rounding
    behind, so ``err_bound`` grows while the window slides. Once it
    reaches ``TRUST_MARGIN`` of the variance itself, the running value
    no longer carries the digits a centered scan would produce.

    Args:
        variance: Running variance of the window.
        err_bound: Accumulated bound on the error of the sum of squares.
        arg: Window length.

    Returns:
        ``1`` when the window has to be scanned again, otherwise ``0``.
    """
    if variance < 0.0:
        return 1
    if err_bound > TRUST_MARGIN * arg * variance:
        return 1
    return 0


def scan_sums(  # pragma: no cover
    stack: Any, row: int, begin: int, end: int, offset: float
) -> tuple[float, float]:
    """Rebuild one window's running sums from the samples.

    Args:
        stack: Column-length workspace.
        row: Stack row that holds the series.
        begin: Inclusive start index.
        end: Exclusive stop index.
        offset: Shift subtracted from every sample.

    Returns:
        The sum and sum of squares of the shifted samples.
    """
    total = 0.0
    squares = 0.0
    for index in range(begin, end):
        value = float(stack[row, index]) - offset
        total += value
        squares += value * value
    return total, squares


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


def scan_pair_sums(  # pragma: no cover
    stack: Any,
    left_row: int,
    right_row: int,
    begin: int,
    end: int,
    off_x: float,
    off_y: float,
) -> tuple[float, float, float, float, float]:
    """Rebuild one pair window's running sums from the samples.

    Args:
        stack: Column-length workspace.
        left_row: Left-series stack row.
        right_row: Right-series stack row.
        begin: Inclusive start index.
        end: Exclusive stop index.
        off_x: Shift subtracted from the left series.
        off_y: Shift subtracted from the right series.

    Returns:
        Left and right sums, left and right sums of squares, and the
        sum of products, all of the shifted samples.
    """
    total_x = 0.0
    total_y = 0.0
    squares_x = 0.0
    squares_y = 0.0
    products = 0.0
    for index in range(begin, end):
        left = float(stack[left_row, index]) - off_x
        right = float(stack[right_row, index]) - off_y
        total_x += left
        total_y += right
        squares_x += left * left
        squares_y += right * right
        products += left * right
    return total_x, total_y, squares_x, squares_y, products


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

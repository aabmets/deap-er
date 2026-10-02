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

from ..window_moments import SCALE_LIMIT, SQUARES_HIGH, SQUARES_LOW

__all__: list[str] = [
    "scan_inverse",
    "scan_variance",
    "scan_pair_moments",
]


def scan_inverse(  # pragma: no cover
    stack: Any, row: int, begin: int, end: int, mean: float, squares: float
) -> float:
    """Pick the power of two that keeps one window's squares in range.

    Mirrors ``scaled_moments`` on the Python backend: a mean square
    inside ``[SQUARES_LOW, SQUARES_HIGH]`` needs no scaling, and any
    other brings the window's largest deviation near 1.

    Args:
        stack: Column-length workspace.
        row: Stack row that holds the series.
        begin: Inclusive start index.
        end: Exclusive stop index.
        mean: Window mean.
        squares: Unscaled mean square deviation of the window.

    Returns:
        The factor to multiply the deviations by, 1.0 when none is needed.
    """
    if not (squares < SQUARES_LOW or squares > SQUARES_HIGH):
        return 1.0
    peak = 0.0
    for index in range(begin, end):
        peak = max(peak, abs(float(stack[row, index]) - mean))
    exponent = -math.frexp(peak)[1]
    exponent = min(max(exponent, -SCALE_LIMIT), SCALE_LIMIT)
    return math.ldexp(1.0, exponent)


def scan_variance(  # pragma: no cover
    stack: Any, row: int, begin: int, end: int
) -> tuple[float, float]:
    """Take one window's variance around its own mean.

    Centering on the window mean is what the Python backend does, so
    the two agree without either of them having to round alike. A
    window whose squares would overflow or underflow is taken again
    with its deviations scaled by a power of two (see ``scan_inverse``).

    Args:
        stack: Column-length workspace.
        row: Stack row that holds the series.
        begin: Inclusive start index.
        end: Exclusive stop index.

    Returns:
        The scaled population variance of the window and the scale;
        the true variance is the first times the square of the second.
    """
    count = float(end - begin)
    total = 0.0
    for index in range(begin, end):
        total += float(stack[row, index])
    mean = total / count
    inverse = 1.0
    residue = 0.0
    squares = 0.0
    for _ in range(2):
        residue = 0.0
        squares = 0.0
        for index in range(begin, end):
            deviation = (float(stack[row, index]) - mean) * inverse
            residue += deviation
            squares += deviation * deviation
        residue /= count
        squares /= count
        if inverse != 1.0:
            break
        inverse = scan_inverse(stack, row, begin, end, mean, squares)
        if inverse == 1.0:
            break
    return squares - residue * residue, 1.0 / inverse


def scan_pair_moments(  # pragma: no cover
    stack: Any, left_row: int, right_row: int, begin: int, end: int
) -> tuple[float, float, float, float, float]:
    """Take one pair window's moments around their own means.

    Each series is scaled on its own, as in ``scan_variance``.

    Args:
        stack: Column-length workspace.
        left_row: Left-series stack row.
        right_row: Right-series stack row.
        begin: Inclusive start index.
        end: Exclusive stop index.

    Returns:
        The scaled covariance, the two scaled variances, and the left
        and right scales of the window. The true covariance is the
        scaled one times both scales, and a true variance is its
        scaled one times its scale squared.
    """
    count = float(end - begin)
    mean_x = 0.0
    mean_y = 0.0
    for index in range(begin, end):
        mean_x += float(stack[left_row, index])
        mean_y += float(stack[right_row, index])
    mean_x /= count
    mean_y /= count
    inverse_x = 1.0
    inverse_y = 1.0
    residue_x = 0.0
    residue_y = 0.0
    squares_x = 0.0
    squares_y = 0.0
    products = 0.0
    for _ in range(2):
        residue_x = 0.0
        residue_y = 0.0
        squares_x = 0.0
        squares_y = 0.0
        products = 0.0
        for index in range(begin, end):
            dev_x = (float(stack[left_row, index]) - mean_x) * inverse_x
            dev_y = (float(stack[right_row, index]) - mean_y) * inverse_y
            residue_x += dev_x
            residue_y += dev_y
            squares_x += dev_x * dev_x
            squares_y += dev_y * dev_y
            products += dev_x * dev_y
        residue_x /= count
        residue_y /= count
        squares_x /= count
        squares_y /= count
        products /= count
        if inverse_x != 1.0 or inverse_y != 1.0:
            break
        inverse_x = scan_inverse(stack, left_row, begin, end, mean_x, squares_x)
        inverse_y = scan_inverse(stack, right_row, begin, end, mean_y, squares_y)
        if inverse_x == 1.0 and inverse_y == 1.0:
            break
    return (
        products - residue_x * residue_y,
        squares_x - residue_x * residue_x,
        squares_y - residue_y * residue_y,
        1.0 / inverse_x,
        1.0 / inverse_y,
    )

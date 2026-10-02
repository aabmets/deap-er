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
    "scan_variance",
    "scan_pair_moments",
]


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

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
import numpy

__all__: list[str] = [
    "SQUARES_LOW",
    "SQUARES_HIGH",
    "SCALE_LIMIT",
    "window_deviations",
    "scaled_moments",
]

SQUARES_LOW = 2.0**-500
SQUARES_HIGH = 2.0**500
SCALE_LIMIT = 1000


def window_deviations(block: numpy.ndarray, length: int) -> numpy.ndarray:
    """Center each window of a strided block on its own mean.

    Subtracting the window mean before squaring is what keeps the
    moments accurate: ``E[x^2] - mean^2`` would cancel away the digits
    the variance is made of whenever the samples sit far from zero.

    A window holding an infinity has no finite mean, so its deviations
    are ``nan`` by definition rather than by accident.

    Args:
        block: Strided view of shape ``(rows, length)``.
        length: Window length.

    Returns:
        The deviations from each window's mean.
    """
    with numpy.errstate(invalid="ignore", over="ignore"):
        mean = numpy.add.reduce(block, axis=-1) / length
        return block - mean[:, None]


def scaled_moments(
    deviations: numpy.ndarray, length: int
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Take the mean deviation and mean square of each window, safely scaled.

    A row whose mean square leaves ``[SQUARES_LOW, SQUARES_HIGH]``
    would overflow or underflow once squared, so it is multiplied by a
    power of two that brings its largest deviation near 1, in place,
    and its moments are taken again. Powers of two scale exactly, so
    every other row keeps its bits and a scaled row's moments are the
    true ones times ``scale ** -1`` and ``scale ** -2``.

    Args:
        deviations: Window deviations of shape ``(rows, length)``.
            Rescaled rows are overwritten.
        length: Window length.

    Returns:
        The mean deviation, the mean square, and the scale of each row.
    """
    with numpy.errstate(invalid="ignore", over="ignore", under="ignore"):
        residue = numpy.add.reduce(deviations, axis=-1) / length
        squares = numpy.add.reduce(deviations * deviations, axis=-1) / length
        extreme = (squares < SQUARES_LOW) | (squares > SQUARES_HIGH)
        scale = numpy.ones_like(squares)
        if extreme.any():
            peak = numpy.max(numpy.abs(deviations[extreme]), axis=-1)
            exponent = numpy.clip(-numpy.frexp(peak)[1], -SCALE_LIMIT, SCALE_LIMIT)
            inverse = numpy.ldexp(1.0, exponent)
            rows = deviations[extreme] * inverse[:, None]
            deviations[extreme] = rows
            residue[extreme] = numpy.add.reduce(rows, axis=-1) / length
            squares[extreme] = numpy.add.reduce(rows * rows, axis=-1) / length
            scale[extreme] = 1.0 / inverse
    return residue, squares, scale

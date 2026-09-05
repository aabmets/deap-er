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

from deap_er.private.various.rng import rng

__all__: list[str] = ["change_peaks"]


def _change_shape(
    axis: list[float], axis_min: float, axis_max: float, sev: float, idx: int
) -> None:
    """Nudge one peak attribute, reflecting it off its bounds.

    Args:
        axis: Per-peak values, modified in place.
        axis_min: Lower bound of the attribute.
        axis_max: Upper bound of the attribute.
        sev: Standard deviation of the change.
        idx: Position of the peak to change.
    """
    change = rng.gauss(0, 1) * sev
    new_value = change + axis[idx]
    if new_value < axis_min:
        axis[idx] = 2.0 * axis_min - axis[idx] - change
    elif new_value > axis_max:
        axis[idx] = 2.0 * axis_max - axis[idx] - change
    else:
        axis[idx] = new_value


def _remove_peaks(landscape: Any, count: int) -> None:
    """Drop ``count`` randomly chosen peaks.

    Args:
        landscape: Moving-peaks instance to modify.
        count: Number of peaks to remove.
    """
    for _ in range(count):
        idx = rng.randrange(len(landscape.peaks_function))
        landscape.peaks_function.pop(idx)
        landscape.peaks_position.pop(idx)
        landscape.peaks_height.pop(idx)
        landscape.peaks_width.pop(idx)
        landscape.last_change_vector.pop(idx)


def _add_peaks(landscape: Any, count: int) -> None:
    """Append ``count`` randomly placed peaks.

    Args:
        landscape: Moving-peaks instance to modify.
        count: Number of peaks to add.
    """
    for _ in range(count):
        rand = rng.choice(landscape.pfunc_pool)
        landscape.peaks_function.append(rand)
        rand = [rng.uniform(landscape.min_coord, landscape.max_coord) for _ in range(landscape.dim)]
        landscape.peaks_position.append(rand)
        rand = rng.uniform(landscape.min_height, landscape.max_height)
        landscape.peaks_height.append(rand)
        rand = rng.uniform(landscape.min_width, landscape.max_width)
        landscape.peaks_width.append(rand)
        rand = [rng.random() - 0.5 for _ in range(landscape.dim)]
        landscape.last_change_vector.append(rand)


def _change_peak_count(landscape: Any) -> None:
    """Add or remove peaks, staying within the configured bounds.

    Args:
        landscape: Moving-peaks instance to modify.
    """
    if landscape.min_peaks is None or landscape.max_peaks is None:
        return

    n_peaks = len(landscape.peaks_function)
    u = rng.random()
    r = landscape.max_peaks - landscape.min_peaks
    if u < 0.5:
        u = rng.random()
        runs = int(round(r * u * landscape.number_severity))
        _remove_peaks(landscape, min(n_peaks - landscape.min_peaks, runs))
    else:
        u = rng.random()
        runs = int(round(r * u * landscape.number_severity))
        _add_peaks(landscape, min(landscape.max_peaks - n_peaks, runs))


def _move_peak(landscape: Any, index: int) -> None:
    """Shift one peak, reflecting it off the coordinate bounds.

    The step is a blend of a fresh random direction and the peak's
    previous direction, controlled by ``lambda_``.

    Args:
        landscape: Moving-peaks instance to modify.
        index: Position of the peak to move.
    """
    len_ = len(landscape.peaks_position[index])
    shift = [rng.random() - 0.5 for _ in range(len_)]
    shift_length = sum(s**2 for s in shift)
    shift_length = landscape.move_severity / math.sqrt(shift_length) if shift_length > 0 else 0

    zipper = zip(shift, landscape.last_change_vector[index], strict=False)
    shift = [shift_length * (1.0 - landscape.lamb) * s + landscape.lamb * c for s, c in zipper]
    shift_length = sum(s**2 for s in shift)
    shift_length = landscape.move_severity / math.sqrt(shift_length) if shift_length > 0 else 0

    shift = [s * shift_length for s in shift]

    new_position = []
    final_shift = []
    for pp, s in zip(landscape.peaks_position[index], shift, strict=False):
        new_coord = pp + s
        if new_coord < landscape.min_coord:
            new_position.append(2.0 * landscape.min_coord - pp - s)
            final_shift.append(-1.0 * s)
        elif new_coord > landscape.max_coord:
            new_position.append(2.0 * landscape.max_coord - pp - s)
            final_shift.append(-1.0 * s)
        else:
            new_position.append(new_coord)
            final_shift.append(s)

    landscape.peaks_position[index] = new_position
    landscape.last_change_vector[index] = final_shift


def change_peaks(landscape: Any) -> None:
    """Change the position, height, width, and number of peaks.

    Args:
        landscape: Moving-peaks instance to modify.
    """
    landscape._optimum = None

    _change_peak_count(landscape)

    for i in range(len(landscape.peaks_function)):
        _move_peak(landscape, i)
        _change_shape(
            landscape.peaks_height,
            landscape.min_height,
            landscape.max_height,
            landscape.height_severity,
            i,
        )
        _change_shape(
            landscape.peaks_width,
            landscape.min_width,
            landscape.max_width,
            landscape.width_severity,
            i,
        )

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

import moocore
import numpy

from deap_er.base.dtypes import Individual

__all__ = ["hypervolume"]


def _minimized_points(population: list[Any]) -> numpy.ndarray:
    """Return objective rows in minimization space (``-wvalues``).

    Args:
        population: Individuals with a Fitness attribute.

    Returns:
        A 2-D array of points, or shape ``(0, 0)`` when ``population``
        is empty.
    """
    if not population:
        return numpy.empty((0, 0), dtype=float)
    return numpy.array([ind.fitness.wvalues for ind in population], dtype=float) * -1


def _has_fitness(obj: object) -> bool:
    """Return whether ``obj`` looks like an individual with Fitness.

    Args:
        obj: Value to inspect.

    Returns:
        True if ``obj`` has a ``fitness`` attribute.
    """
    return hasattr(obj, "fitness")


def hypervolume(
    points: numpy.ndarray | list[Individual] | Individual,
    ref_point: numpy.ndarray | list[float] | None = None,
) -> float:
    """Return the hypervolume of a point set or a population.

    Minimization is assumed. An individual or a sequence of
    individuals is converted via ``-wvalues``. A bare point matrix
    (no ``fitness``) is used as-is. ``ref_point`` is in that same
    space. Delegates to ``moocore.hypervolume``.

    Args:
        points: Minimized objective rows, one individual, or a
            population with Fitness.
        ref_point: Reference point. Optional. If omitted, the worst
            value of each objective plus one is used.

    Returns:
        The hypervolume of the point set.
    """
    if _has_fitness(points):
        arr = _minimized_points([points])
    elif not isinstance(points, numpy.ndarray) and points and _has_fitness(points[0]):
        arr = _minimized_points(list(points))
    else:
        arr = numpy.asarray(points, dtype=float)
    if arr.size == 0:
        return 0.0
    ref = numpy.max(arr, axis=0) + 1 if ref_point is None else numpy.asarray(ref_point)
    return float(moocore.hypervolume(arr, ref=ref, maximise=False))

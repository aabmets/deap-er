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
from collections.abc import Callable
from typing import Any

import numpy

from deap_er.base.dtypes import Individual

from .hypervolume import HyperVolume

__all__ = ["least_contrib"]


def _compute_hv(data: tuple[numpy.ndarray, numpy.ndarray]) -> float:
    """Compute the hypervolume of one point set against a reference point.

    Args:
        data: Pair of ``(point_set, ref_point)``.

    Returns:
        The hypervolume of ``point_set``.
    """
    point_set, ref_point = data[0], data[1]
    hv = HyperVolume(ref_point)
    return hv.compute(point_set)


def least_contrib(
    population: list[Individual],
    ref_point: list[float] | numpy.ndarray | None = None,
    map_func: Callable[..., Any] | None = map,
) -> int | numpy.ndarray:
    """Return the index of the individual with the least hypervolume contribution.

    Minimization is implicitly assumed.

    Args:
        population: Non-dominated individuals, each with a Fitness
            attribute.
        ref_point: Reference point for the hypervolume. Optional. If
            omitted, the worst value of each objective plus one is
            used.
        map_func: Map that applies a callable to an iterable.
            Optional. A pool map can be supplied to parallelize the
            per-individual computations. Defaults to the built-in
            single-process ``map``.

    Returns:
        The index of the individual with the least hypervolume
        contribution.
    """
    wvals = numpy.array([ind.fitness.wvalues for ind in population]) * -1
    point = numpy.max(wvals, axis=0) + 1 if ref_point is None else numpy.array(ref_point)

    data = []
    for i in range(len(population)):
        point_set = numpy.concatenate((wvals[:i], wvals[i + 1 :]))
        data.append((point_set, point))

    mapper = map if map_func is None else map_func
    contrib_values = list(mapper(_compute_hv, data))
    return int(numpy.argmax(contrib_values))

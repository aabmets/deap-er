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
from itertools import chain

import numpy
from numpy import ndarray

from deap_er.base.typedefs import Individual
from deap_er.utilities.sorting import sort_non_dominated

from ._nsga_3 import (
    _associate_to_niche,
    _find_extreme_points,
    _find_intercepts,
    _select_from_niche,
)

__all__ = ["sel_nsga_3", "SelNSGA3WithMemory"]


class SelNSGA3WithMemory:
    """NSGA-III selection that remembers ideal, nadir, and extreme points.

    Instances can be registered into a Toolbox.

    Args:
        ref_points: Reference points for selection.
    """

    def __init__(self, ref_points: ndarray) -> None:
        """See the class docstring."""
        self.ref_points = ref_points
        self.best_point = numpy.full((1, ref_points.shape[1]), numpy.inf)
        self.worst_point = numpy.full((1, ref_points.shape[1]), -numpy.inf)
        self.extreme_points: ndarray | None = None

    def __call__(self, individuals: list[Individual], sel_count: int) -> list[Individual]:
        """Select individuals for the next generation.

        Args:
            individuals: Individuals to select from.
            sel_count: Number of individuals to select.

        Returns:
            The selected individuals.
        """
        chosen = sel_nsga_3(
            individuals,
            sel_count,
            self.ref_points,
            self.best_point,
            self.worst_point,
            self.extreme_points,
            self,
        )
        return chosen


def sel_nsga_3(
    individuals: list[Individual],
    sel_count: int,
    ref_points: ndarray,
    best_point: ndarray | None = None,
    worst_point: ndarray | None = None,
    extreme_points: ndarray | None = None,
    _memory: SelNSGA3WithMemory | None = None,
) -> list[Individual]:
    """Select the next generation with NSGA-III.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.
        ref_points: Reference points used for niche selection.
        best_point: Ideal point of the previous generation. If
            omitted, it is taken from the current individuals.
        worst_point: Nadir point of the previous generation. If
            omitted, it is taken from the current individuals.
        extreme_points: Extreme points of the previous generation.
            If omitted, they are taken from the current individuals.
        _memory: ``SelNSGA3WithMemory`` instance that stores the
            updated ideal, nadir, and extreme points. Not intended
            for manual use.

    Returns:
        The selected individuals.
    """
    if not individuals or sel_count <= 0:
        return []
    pareto_fronts = sort_non_dominated(individuals, sel_count)

    fitness = numpy.array([ind.fitness.wvalues for f in pareto_fronts for ind in f])
    fitness *= -1

    if best_point is not None and worst_point is not None:
        best_point = numpy.min(numpy.concatenate((fitness, best_point), axis=0), axis=0)
        worst_point = numpy.max(numpy.concatenate((fitness, worst_point), axis=0), axis=0)
    else:
        best_point = numpy.min(fitness, axis=0)
        worst_point = numpy.max(fitness, axis=0)

    extreme_points = _find_extreme_points(fitness, best_point, extreme_points)
    front_worst = numpy.max(fitness[: sum(len(f) for f in pareto_fronts), :], axis=0)
    intercepts = _find_intercepts(extreme_points, best_point, worst_point, front_worst)
    niches, dist = _associate_to_niche(fitness, ref_points, best_point, intercepts)

    niche_counts = numpy.zeros(len(ref_points), dtype=numpy.int64)
    index, counts = numpy.unique(niches[: -len(pareto_fronts[-1])], return_counts=True)
    niche_counts[index] = counts

    chosen = list(chain(*pareto_fronts[:-1]))
    selected = len(chosen)
    selected = _select_from_niche(
        pareto_fronts[-1], sel_count - selected, niches[selected:], dist[selected:], niche_counts
    )
    chosen.extend(selected)

    if isinstance(_memory, SelNSGA3WithMemory):
        _memory.best_point = numpy.asarray(best_point).reshape((1, -1))
        _memory.worst_point = numpy.asarray(worst_point).reshape((1, -1))
        _memory.extreme_points = extreme_points

    return chosen

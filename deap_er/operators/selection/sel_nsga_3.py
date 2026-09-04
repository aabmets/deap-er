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
from deap_er.rng import rng
from deap_er.utilities.sorting import sort_non_dominated

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


def _find_extreme_points(
    fitness: ndarray, best_point: ndarray, extreme_points: ndarray | None = None
) -> ndarray:
    """Find one extreme point per objective.

    Previous extreme points, when given, are considered together with
    the current fitness values.

    Args:
        fitness: Objective values of the current fronts.
        best_point: Current ideal point.
        extreme_points: Extreme points of the previous generation.

    Returns:
        One extreme point per objective.
    """
    if extreme_points is not None:
        fitness = numpy.concatenate((fitness, extreme_points), axis=0)

    ft = fitness - best_point
    asf = numpy.eye(best_point.shape[0])
    asf[asf == 0] = 1e6
    asf = numpy.max(ft * asf[:, numpy.newaxis, :], axis=2)

    min_asf_idx = numpy.argmin(asf, axis=1)
    return fitness[min_asf_idx, :]


def _find_intercepts(
    extreme_points: ndarray, best_point: ndarray, current_worst: ndarray, front_worst: ndarray
) -> ndarray:
    """Compute axis intercepts of the hyperplane through the extreme points.

    Falls back to a worst-point estimate when the hyperplane is
    degenerate or the intercepts are not usable.

    Args:
        extreme_points: One extreme point per objective.
        best_point: Current ideal point.
        current_worst: Worst point including memory from prior generations.
        front_worst: Worst point on the current fronts.

    Returns:
        Intercepts used to scale the objectives.
    """
    b = numpy.ones(extreme_points.shape[1])
    big_a = extreme_points - best_point
    try:
        x = numpy.linalg.solve(big_a, b)
    except numpy.linalg.LinAlgError:
        intercepts = current_worst
    else:
        if numpy.count_nonzero(x) != len(x):
            intercepts = front_worst
        else:
            intercepts = 1 / x

            if (
                not numpy.allclose(numpy.dot(big_a, x), b)
                or numpy.any(intercepts <= 1e-6)
                or numpy.any((intercepts + best_point) > current_worst)
            ):
                intercepts = front_worst

    return intercepts


def _associate_to_niche(
    fitness: ndarray, reference_points: ndarray, best_point: ndarray, intercepts: ndarray
) -> tuple[ndarray, ndarray]:
    """Assign each individual to the nearest reference-point niche.

    Args:
        fitness: Objective values of the current fronts.
        reference_points: Reference points that define the niches.
        best_point: Current ideal point.
        intercepts: Axis intercepts used to normalize the objectives.

    Returns:
        Niche index and distance to that niche for each individual.
    """
    fn = (fitness - best_point) / (intercepts - best_point)
    fn = fn[:, numpy.newaxis, :]
    norm = numpy.linalg.norm(reference_points, axis=1)
    distances = numpy.sum(fn * reference_points, axis=2) / norm.reshape(1, -1)
    unit = reference_points / norm[:, numpy.newaxis]
    distances = numpy.linalg.norm(distances[:, :, numpy.newaxis] * unit - fn, axis=2)

    niches = numpy.argmin(distances, axis=1)
    distances = distances[numpy.arange(niches.shape[0]), niches]
    return niches, distances


def _select_from_niche(
    individuals: list[Individual],
    count: int,
    niches: ndarray,
    distances: ndarray,
    niche_counts: ndarray,
) -> list[Individual]:
    """Fill the remaining slots from the last front by niche.

    Prefers under-represented niches. An empty niche takes its closest
    member; a non-empty niche takes a random remaining member.

    Args:
        individuals: Individuals on the last accepted front.
        count: Number of individuals still needed.
        niches: Niche index of each individual on that front.
        distances: Distance of each individual to its niche.
        niche_counts: Current occupancy of each niche. Updated in place.

    Returns:
        Individuals chosen from the last front.
    """
    selected = []
    available = numpy.ones(len(individuals), dtype=numpy.bool)
    while len(selected) < count:
        n = count - len(selected)

        available_niches = numpy.zeros(len(niche_counts), dtype=numpy.bool)
        available_niches[numpy.unique(niches[available])] = True
        min_count = numpy.min(niche_counts[available_niches])

        logical_and = numpy.logical_and(available_niches, niche_counts == min_count)
        selected_niches = numpy.flatnonzero(logical_and)
        rng.shuffle(selected_niches)
        selected_niches = selected_niches[:n]

        for niche in selected_niches:
            logical_and = numpy.logical_and(niches == niche, available)
            niche_individuals = numpy.flatnonzero(logical_and)
            rng.shuffle(niche_individuals)

            if niche_counts[niche] == 0:
                arg_min = numpy.argmin(distances[niche_individuals])
                sel_index = niche_individuals[arg_min]
            else:
                sel_index = niche_individuals[0]

            available[sel_index] = False
            niche_counts[niche] += 1
            selected.append(individuals[sel_index])

    return selected

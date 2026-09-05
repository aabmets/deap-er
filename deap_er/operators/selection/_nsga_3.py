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
from numpy import ndarray

from deap_er.base.typedefs import Individual
from deap_er.rng import rng

__all__: list[str] = []


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

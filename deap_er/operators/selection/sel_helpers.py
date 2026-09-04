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

from deap_er.base.dtypes import Individual

__all__ = ["assign_crowding_dist", "uniform_reference_points"]


def assign_crowding_dist(individuals: list[Individual]) -> None:
    """Assign a crowding distance to each individual's fitness.

    The distance is stored on the ``crowding_dist`` attribute of each
    individual's fitness. The individuals are modified in place.

    Args:
        individuals: Individuals with Fitness attributes.
    """
    if len(individuals) == 0:
        return

    distances = [0.0] * len(individuals)
    crowd = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]
    n_obj = len(individuals[0].fitness.values)

    for i in range(n_obj):
        crowd.sort(key=lambda element, obj_i=i: element[0][obj_i])
        distances[crowd[0][1]] = float("inf")
        distances[crowd[-1][1]] = float("inf")
        if crowd[-1][0][i] == crowd[0][0][i]:
            continue
        norm = n_obj * float(crowd[-1][0][i] - crowd[0][0][i])
        for prev, cur, next_ in zip(crowd[:-2], crowd[1:-1], crowd[2:], strict=False):
            distances[cur[1]] += (next_[0][i] - prev[0][i]) / norm

    for i, dist in enumerate(distances):
        individuals[i].fitness.crowding_dist = dist


def uniform_reference_points(
    objectives: int, ref_ppo: int = 4, scaling: float | None = None
) -> numpy.ndarray:
    """Generate reference points uniformly on the unit simplex.

    Points lie on the hyperplane that intersects each axis at 1.
    ``scaling`` shrinks that layer toward the simplex center so
    several layers can be combined.

    Args:
        objectives: Number of objectives.
        ref_ppo: Number of reference points per objective.
        scaling: Optional scaling factor for combining layers.

    Returns:
        Uniform reference points.
    """

    def _recursive(
        ref: numpy.ndarray, ovs: int, left: int, total: int, depth: int
    ) -> list[numpy.ndarray]:
        """Fill remaining objectives of one Das-Dennis reference point.

        Args:
            ref: Partial weight vector being filled.
            ovs: Number of objectives.
            left: Remaining integer budget to distribute.
            total: Total budget that sets the simplex spacing.
            depth: Objective index currently being assigned.

        Returns:
            Completed reference points generated from this prefix.
        """
        points = []
        if depth == ovs - 1:
            ref[depth] = left / total
            points.append(ref)
        else:
            for i in range(left + 1):
                ref[depth] = i / total
                rc = ref.copy()
                li = left - i
                d1 = depth + 1
                result = _recursive(rc, ovs, li, total, d1)
                points.extend(result)
        return points

    zeros = numpy.zeros(objectives)
    ref_points = _recursive(zeros, objectives, ref_ppo, ref_ppo, 0)
    ref_points = numpy.array(ref_points)

    if scaling is not None:
        ref_points *= scaling
        ref_points += (1 - scaling) / objectives

    return ref_points

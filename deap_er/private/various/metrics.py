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
from __future__ import annotations

from math import hypot
from typing import TYPE_CHECKING, Any

import numpy
from scipy import spatial

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = ["nsga_diversity", "nsga_convergence", "inv_gen_dist", "duplicate_count"]


def _objective_row(point: Any) -> tuple[Any, ...]:
    """Return objective values from an individual or a raw vector.

    Args:
        point: An individual with ``fitness.values``, or a sequence of
            objective coordinates.

    Returns:
        The fitness values when present and non-empty, otherwise
        ``tuple(point)``.
    """
    fitness = getattr(point, "fitness", None)
    if fitness is not None:
        values = getattr(fitness, "values", ())
        if values:
            return tuple(values)
    return tuple(point)


def duplicate_count(population: list[Any], key: Any | None = None) -> int:
    """Return how many individuals are duplicates of an earlier one.

    Hashable keys use a set scan in ``O(n)``. Unhashable but sortable keys
    use a sort in ``O(n log n)``. Keys that are neither hashable nor
    mutually sortable fall back to list membership in ``O(n^2)``.
    A NumPy array key is hashed by shape, dtype, and bytes so ndarray
    individuals take the set path.

    Hashable keys must satisfy Python's hash/equality contract: equal keys
    must hash equally. The set path counts by hash bucket; equal keys with
    unequal hashes are treated as distinct (unlike a pure ``==`` scan).

    Args:
        population: Individuals to scan.
        key: Extracts the compared value. Defaults to the identity.

    Returns:
        ``len(population)`` minus the number of distinct keys.
    """
    extract = key if key is not None else (lambda obj: obj)
    if not population:
        return 0
    keys = []
    for item in population:
        value = extract(item)
        if isinstance(value, numpy.ndarray):
            value = (value.shape, str(value.dtype), value.tobytes())
        keys.append(value)
    try:
        return len(keys) - len(set(keys))
    except TypeError:
        pass
    try:
        ordered = sorted(keys)
    except TypeError:
        unique: list[Any] = []
        for value in keys:
            if value not in unique:
                unique.append(value)
        return len(keys) - len(unique)
    distinct = 1
    for index in range(1, len(ordered)):
        if ordered[index] != ordered[index - 1]:
            distinct += 1
    return len(keys) - distinct


def nsga_diversity(population: list[Individual], first: Individual, last: Individual) -> float:
    """Return the NSGA-II diversity metric of a Pareto front.

    ``population`` is the front to score. ``first`` and ``last`` are
    the extreme points of the optimal Pareto front, as in Deb's
    original NSGA-II article. Each extreme may be an individual
    (objectives from ``fitness.values``) or a raw objective vector.
    Smaller values indicate better spread.

    Args:
        population: Pareto front to evaluate.
        first: First extreme point of the optimal Pareto front.
        last: Last extreme point of the optimal Pareto front.

    Returns:
        The diversity metric of the front.
    """
    ordered = sorted(population, key=lambda ind: ind.fitness.values[0])
    start = _objective_row(first)
    end = _objective_row(last)
    df = hypot(ordered[0].fitness.values[0] - start[0], ordered[0].fitness.values[1] - start[1])
    dl = hypot(ordered[-1].fitness.values[0] - end[0], ordered[-1].fitness.values[1] - end[1])

    def fn(f_: Individual, s_: Individual) -> float:
        return hypot(
            f_.fitness.values[0] - s_.fitness.values[0], f_.fitness.values[1] - s_.fitness.values[1]
        )

    zipper = zip(ordered[:-1], ordered[1:], strict=False)
    dt = [fn(first, second) for first, second in zipper]

    if len(ordered) == 1:
        return 1.0

    dm = sum(dt) / len(dt)
    di = sum(abs(d_i - dm) for d_i in dt)
    denom = df + dl + len(dt) * dm
    if denom <= 0:
        return 1.0
    return (df + dl + di) / denom


def nsga_convergence(population: list[Individual], optimal: list[Individual]) -> float:
    """Return the NSGA-II convergence metric of a Pareto front.

    ``population`` is the front to score and ``optimal`` is the true
    Pareto front, as in Deb's original NSGA-II article. Each reference
    point may be an individual or a raw objective vector. Smaller
    values indicate closer solutions.

    Args:
        population: Pareto front to evaluate.
        optimal: Optimal Pareto front.

    Returns:
        The convergence metric of the front.
    """
    front = numpy.asarray([ind.fitness.values for ind in population], dtype=float)
    truth = numpy.asarray([_objective_row(opt) for opt in optimal], dtype=float)
    minima = numpy.min(spatial.distance.cdist(front, truth), axis=1)
    return float(numpy.mean(minima))


def inv_gen_dist(ind1: Individual, ind2: Individual) -> Any:
    """Compute the inverted generational distance between two point sets.

    IGD measures how well one approximation covers another in
    multi-objective optimization.

    Args:
        ind1: First point set.
        ind2: Second point set.

    Returns:
        The average distance from each point in ``ind2`` to the
        nearest point in ``ind1``.
    """
    first = numpy.asarray([_objective_row(point) for point in ind1], dtype=float)
    second = numpy.asarray([_objective_row(point) for point in ind2], dtype=float)
    distances = spatial.distance.cdist(first, second)
    minima = numpy.min(distances, axis=0)
    return numpy.average(minima)

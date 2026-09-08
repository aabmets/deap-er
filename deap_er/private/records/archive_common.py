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

import math
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy

from deap_er.private.fitness import has_comparable_fitness
from deap_er.private.various.rng import rng

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = [
    "ArchiveStats",
    "MapElitesArchive",
    "check_archive_add",
    "make_archive_stats",
    "nearest_index",
    "sample_random_elites",
]


@dataclass(frozen=True, slots=True)
class ArchiveStats:
    """Summary statistics for a MAP-Elites archive.

    Attributes:
        num_elites: Number of stored elites.
        num_cells: Capacity of the tessellation, or the current elite
            count when the archive has no fixed cell budget.
        coverage: ``num_elites / num_cells``, or 0 when ``num_cells``
            is 0.
        qd_score: Sum of the first weighted objective over elites.
            Requires single-objective fitness on every stored elite.
    """

    num_elites: int
    num_cells: int
    coverage: float
    qd_score: float


class MapElitesArchive(Protocol):
    """``add`` / ``random_elites`` / ``stats`` surface for MAP-Elites.

    :meta private:
    """

    def add(self, individual: Any, descriptor: Sequence[float]) -> bool:
        """Insert ``individual`` when it improves the archive."""

    def random_elites(self, n: int, *, replace: bool = True) -> list[Individual]:
        """Sample stored elites."""

    @property
    def stats(self) -> ArchiveStats:
        """Coverage and quality-diversity score."""

    def __len__(self) -> int:
        """Return the number of stored elites."""


def check_archive_add(
    individual: Any,
    descriptor: Sequence[float] | numpy.ndarray,
    dimensions: int,
    archive_name: str,
) -> bool:
    """Return whether ``individual`` is admissible for ``add``.

    Args:
        individual: Candidate with a fitness attribute.
        descriptor: Continuous behavior coordinates.
        dimensions: Expected descriptor length.
        archive_name: Class name used in the multi-objective error.

    Returns:
        True when the candidate may compete for a niche.

    Raises:
        ValueError: If ``descriptor`` length does not match
            ``dimensions``, or ``fitness`` is not single-objective.
    """
    if len(descriptor) != dimensions:
        raise ValueError(
            f"descriptor length {len(descriptor)} does not match {dimensions} dimensions"
        )
    if not all(math.isfinite(float(value)) for value in descriptor):
        return False
    if not has_comparable_fitness(individual):
        return False
    if len(individual.fitness.weights) != 1:
        raise ValueError(f"{archive_name} requires single-objective fitness")
    return True


def make_archive_stats(elites: Iterable[Individual], num_cells: int) -> ArchiveStats:
    """Build :class:`ArchiveStats` from stored elites.

    Args:
        elites: Individuals currently in the archive.
        num_cells: Tessellation size or elite budget used for coverage.

    Returns:
        Coverage and quality-diversity totals.
    """
    qd_score = 0.0
    num_elites = 0
    for individual in elites:
        num_elites += 1
        if individual.fitness.is_valid():
            qd_score += individual.fitness.wvalues[0]
    coverage = num_elites / num_cells if num_cells else 0.0
    return ArchiveStats(
        num_elites=num_elites,
        num_cells=num_cells,
        coverage=coverage,
        qd_score=qd_score,
    )


def nearest_index(points: numpy.ndarray, descriptor: Sequence[float] | numpy.ndarray) -> int:
    """Return the row of ``points`` nearest to ``descriptor``.

    Args:
        points: ``(n, dims)`` array of stored coordinates.
        descriptor: Query coordinates of length ``dims``.

    Returns:
        Index of the closest row. Ties take the lowest index.
    """
    query = numpy.asarray(descriptor, dtype=numpy.float64)
    delta = points - query
    return int(numpy.square(delta).sum(axis=1).argmin())


def sample_random_elites(
    elites: Sequence[Individual],
    n: int,
    *,
    replace: bool,
    empty_message: str,
) -> list[Individual]:
    """Sample copies of ``elites`` the way MAP-Elites archives do.

    Args:
        elites: Stored individuals.
        n: Number of elites to return.
        replace: Sample with replacement when True.
        empty_message: ``IndexError`` text when ``elites`` is empty.

    Returns:
        Deep copies of the sampled elites.

    Raises:
        IndexError: If ``elites`` is empty and ``n`` is positive.
        ValueError: If ``n`` is negative, or ``replace`` is False and
            ``n`` exceeds the number of elites.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return []
    if not elites:
        raise IndexError(empty_message)
    if not replace and n > len(elites):
        raise ValueError("n exceeds the number of elites when replace is False")
    if replace:
        return [deepcopy(rng.choice(elites)) for _ in range(n)]
    return [deepcopy(individual) for individual in rng.sample(elites, n)]

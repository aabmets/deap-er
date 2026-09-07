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
from collections.abc import Iterator, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy

from deap_er.private.records.archive_common import (
    ArchiveStats,
    check_archive_add,
    make_archive_stats,
    nearest_index,
    sample_random_elites,
)

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = ["UnstructuredArchive"]


class UnstructuredArchive:
    """MAP-Elites archive that keeps elites by descriptor distance.

    A candidate is added when it is at least ``min_distance`` from
    every stored elite and the archive is under capacity. Otherwise it
    replaces the nearest neighbor if it is strictly fitter. Scale
    descriptor axes yourself when units differ (for example turnover
    versus win rate).

    Args:
        dimensions: Length of each behavior descriptor.
        min_distance: Euclidean threshold that opens a new niche.
        max_elites: Optional cap. When full, a far candidate competes
            with the nearest elite instead of growing the archive.
    """

    def __init__(
        self,
        dimensions: int,
        min_distance: float,
        *,
        max_elites: int | None = None,
    ) -> None:
        """See the class docstring."""
        if dimensions < 1:
            raise ValueError("dimensions must be at least 1")
        if not math.isfinite(min_distance) or min_distance <= 0.0:
            raise ValueError("min_distance must be a positive finite number")
        if max_elites is not None and max_elites < 1:
            raise ValueError("max_elites must be at least 1")
        self._dimensions = int(dimensions)
        self._min_distance = float(min_distance)
        self._max_elites = max_elites
        self._elites: list[Individual] = []
        self._descriptors = numpy.empty((0, self._dimensions), dtype=numpy.float64)

    @property
    def dimensions(self) -> int:
        """Number of behavior dimensions."""
        return self._dimensions

    @property
    def min_distance(self) -> float:
        """Euclidean threshold that opens a new niche."""
        return self._min_distance

    @property
    def max_elites(self) -> int | None:
        """Elite cap, or None when the archive may grow without bound."""
        return self._max_elites

    @property
    def descriptors(self) -> numpy.ndarray:
        """Copy of stored descriptors with shape ``(n, dimensions)``."""
        return self._descriptors.copy()

    @property
    def stats(self) -> ArchiveStats:
        """Coverage and quality-diversity score of the archive.

        ``num_cells`` is ``max_elites`` when a cap is set, otherwise
        the current elite count (so uncapped coverage is 1.0 when the
        archive is non-empty). ``qd_score`` is the sum of
        ``fitness.wvalues[0]`` over elites.
        """
        num_cells = self._max_elites if self._max_elites is not None else len(self._elites)
        return make_archive_stats(self._elites, num_cells)

    def add(self, individual: Any, descriptor: Sequence[float]) -> bool:
        """Insert ``individual`` when it opens a niche or beats a neighbor.

        Args:
            individual: Candidate with a valid fitness attribute.
            descriptor: Continuous behavior coordinates.

        Returns:
            True when the archive stores ``individual``.

        Raises:
            ValueError: If ``descriptor`` length does not match
                ``dimensions``, or ``fitness`` is not single-objective.
        """
        if not check_archive_add(individual, descriptor, self.dimensions, "UnstructuredArchive"):
            return False
        query = numpy.asarray(descriptor, dtype=numpy.float64)
        if not self._elites:
            self._elites.append(deepcopy(individual))
            self._descriptors = query.reshape(1, -1)
            return True
        nearest = nearest_index(self._descriptors, query)
        dist = float(numpy.linalg.norm(self._descriptors[nearest] - query))
        at_capacity = self._max_elites is not None and len(self._elites) >= self._max_elites
        if dist >= self._min_distance and not at_capacity:
            self._elites.append(deepcopy(individual))
            self._descriptors = numpy.vstack((self._descriptors, query))
            return True
        incumbent = self._elites[nearest]
        if individual.fitness <= incumbent.fitness:
            return False
        self._elites[nearest] = deepcopy(individual)
        self._descriptors[nearest] = query
        return True

    def elite_at(self, descriptor: Sequence[float]) -> Individual | None:
        """Return the nearest stored elite to ``descriptor``.

        Args:
            descriptor: Continuous behavior coordinates.

        Returns:
            The nearest elite, or None when the archive is empty or
            ``descriptor`` is non-finite.

        Raises:
            ValueError: If ``descriptor`` length does not match
                ``dimensions``.
        """
        if len(descriptor) != self.dimensions:
            raise ValueError(
                f"descriptor length {len(descriptor)} does not match {self.dimensions} dimensions"
            )
        if not all(math.isfinite(float(value)) for value in descriptor):
            return None
        if not self._elites:
            return None
        return self._elites[nearest_index(self._descriptors, descriptor)]

    def random_elites(self, n: int, *, replace: bool = True) -> list[Individual]:
        """Sample elites uniformly from stored members.

        Args:
            n: Number of elites to return.
            replace: Sample with replacement when True.

        Returns:
            Stored elites from distinct or repeated members.

        Raises:
            IndexError: If the archive is empty.
            ValueError: If ``n`` is negative, or ``replace`` is False and
                ``n`` exceeds the number of elites.
        """
        return sample_random_elites(
            self._elites,
            n,
            replace=replace,
            empty_message="random_elites from empty UnstructuredArchive",
        )

    def clear(self) -> None:
        """Remove every stored elite."""
        self._elites.clear()
        self._descriptors = numpy.empty((0, self._dimensions), dtype=numpy.float64)

    def __len__(self) -> int:
        """Return the number of stored elites."""
        return len(self._elites)

    def __iter__(self) -> Iterator[Individual]:
        """Iterate over stored elites."""
        return iter(self._elites)

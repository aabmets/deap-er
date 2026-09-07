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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from deap_er.private.records.grid_archive_helpers import (
    descriptor_to_index,
    index_to_descriptor_center,
    parse_grid_config,
)
from deap_er.private.various.rng import rng

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = ["ArchiveStats", "GridArchive"]


@dataclass(frozen=True, slots=True)
class ArchiveStats:
    """Summary statistics for a :class:`GridArchive`.

    Attributes:
        num_elites: Number of filled grid cells.
        num_cells: Total number of cells in the tessellation.
        coverage: Fraction of cells that contain an elite.
        qd_score: Sum of the first weighted objective over elites.
            Requires single-objective fitness on every stored elite.
    """

    num_elites: int
    num_cells: int
    coverage: float
    qd_score: float


class GridArchive:
    """MAP-Elites grid archive indexed by a behavior descriptor.

    The caller supplies a continuous behavior descriptor for each
    individual. The archive bins descriptors into a uniform grid and
    keeps the best individual per cell according to ``fitness``.

    ``fitness`` must be **single-objective** (one weight). Multi-objective
    fitness types are rejected by :meth:`add`. ``stats.qd_score`` sums
    the first weighted objective (:attr:`~deap_er.base.Fitness.wvalues`
    element zero) across filled cells.

    Args:
        ranges: ``(low, high)`` bounds per behavior dimension.
        bins: Resolution per dimension, or one integer for every
            dimension.
    """

    def __init__(
        self,
        ranges: Sequence[tuple[float, float]],
        bins: Sequence[int] | int,
    ) -> None:
        """See the class docstring."""
        self._ranges, self._bins, self._num_cells = parse_grid_config(ranges, bins)
        self._cells: dict[tuple[int, ...], Individual] = {}

    @property
    def dimensions(self) -> int:
        """Number of behavior dimensions."""
        return len(self._ranges)

    @property
    def bins(self) -> tuple[int, ...]:
        """Resolution of the grid along each behavior dimension."""
        return self._bins

    @property
    def ranges(self) -> tuple[tuple[float, float], ...]:
        """``(low, high)`` bounds per behavior dimension."""
        return self._ranges

    @property
    def stats(self) -> ArchiveStats:
        """Coverage and quality-diversity score of the archive.

        ``qd_score`` is the sum of ``fitness.wvalues[0]`` over elites.
        It is a MAP-Elites-style scalar quality total, not a sum across
        multiple objectives.
        """
        qd_score = 0.0
        for individual in self._cells.values():
            if individual.fitness.is_valid():
                qd_score += individual.fitness.wvalues[0]
        num_elites = len(self._cells)
        coverage = num_elites / self._num_cells if self._num_cells else 0.0
        return ArchiveStats(
            num_elites=num_elites,
            num_cells=self._num_cells,
            coverage=coverage,
            qd_score=qd_score,
        )

    def descriptor_to_index(self, descriptor: Sequence[float]) -> tuple[int, ...]:
        """Map a behavior descriptor to its grid cell.

        Coordinates outside ``ranges`` are clipped before binning.

        Args:
            descriptor: Continuous behavior coordinates.

        Returns:
            Integer grid index per dimension.

        Raises:
            ValueError: If ``descriptor`` length does not match
                ``dimensions``.
        """
        return descriptor_to_index(descriptor, self._ranges, self._bins)

    def index_to_descriptor_center(self, index: tuple[int, ...]) -> tuple[float, ...]:
        """Return the center of a grid cell in behavior space.

        Args:
            index: Integer grid index per dimension.

        Returns:
            Center coordinate per behavior dimension.

        Raises:
            ValueError: If ``index`` length or any coordinate is out of
                range.
        """
        return index_to_descriptor_center(index, self._ranges, self._bins)

    def add(self, individual: Any, descriptor: Sequence[float]) -> bool:
        """Insert ``individual`` when it improves its behavior cell.

        Args:
            individual: Candidate with a valid fitness attribute.
            descriptor: Continuous behavior coordinates.

        Returns:
            True when the archive stores ``individual``.

        Raises:
            ValueError: If ``descriptor`` length does not match
                ``dimensions``, or ``fitness`` is not single-objective.
        """
        if len(descriptor) != self.dimensions:
            raise ValueError(
                f"descriptor length {len(descriptor)} does not match {self.dimensions} dimensions"
            )
        if not all(math.isfinite(float(value)) for value in descriptor):
            return False
        if not hasattr(individual, "fitness") or not individual.fitness.is_valid():
            return False
        if len(individual.fitness.weights) != 1:
            raise ValueError("GridArchive requires single-objective fitness")
        if not math.isfinite(float(individual.fitness.wvalues[0])):
            return False
        cell = self.descriptor_to_index(descriptor)
        incumbent = self._cells.get(cell)
        if incumbent is not None and individual.fitness <= incumbent.fitness:
            return False
        self._cells[cell] = deepcopy(individual)
        return True

    def elite_at(self, descriptor: Sequence[float]) -> Individual | None:
        """Return the elite in the cell for ``descriptor``.

        Args:
            descriptor: Continuous behavior coordinates.

        Returns:
            The stored elite, or None when the cell is empty or
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
        return self._cells.get(self.descriptor_to_index(descriptor))

    def get(self, index: tuple[int, ...]) -> Individual | None:
        """Return the elite stored at ``index``.

        Args:
            index: Integer grid index per dimension.

        Returns:
            The stored elite, or None when the cell is empty.
        """
        return self._cells.get(index)

    def random_elites(self, n: int, *, replace: bool = True) -> list[Individual]:
        """Sample elites uniformly from filled cells.

        Args:
            n: Number of elites to return.
            replace: Sample with replacement when True.

        Returns:
            Stored elites from distinct or repeated cells.

        Raises:
            IndexError: If the archive is empty.
            ValueError: If ``n`` is negative, or ``replace`` is False and
                ``n`` exceeds the number of elites.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return []
        elites = list(self._cells.values())
        if not elites:
            raise IndexError("random_elites from empty GridArchive")
        if not replace and n > len(elites):
            raise ValueError("n exceeds the number of elites when replace is False")
        if replace:
            return [deepcopy(rng.choice(elites)) for _ in range(n)]
        return [deepcopy(individual) for individual in rng.sample(elites, n)]

    def clear(self) -> None:
        """Remove every stored elite."""
        self._cells.clear()

    def __len__(self) -> int:
        """Return the number of filled cells."""
        return len(self._cells)

    def __contains__(self, index: tuple[int, ...]) -> bool:
        """Return whether ``index`` holds an elite."""
        return index in self._cells

    def __iter__(self) -> Iterator[Individual]:
        """Iterate over stored elites."""
        return iter(self._cells.values())

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

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy
from scipy.spatial import KDTree

from deap_er.private.records.archive_common import (
    ArchiveStats,
    check_archive_add,
    elite_at_descriptor,
    make_archive_stats,
    nearest_index,
    replace_cell_if_better,
    sample_random_elites,
)
from deap_er.private.records.cvt_centroids import cvt_centroids, parse_centroids

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = ["CvtArchive", "KDTREE_MIN_CENTROIDS", "cvt_centroids"]

KDTREE_MIN_CENTROIDS = 512


class CvtArchive:
    """MAP-Elites archive that assigns descriptors to CVT centroids.

    Each individual is stored in the Voronoi cell of the nearest
    centroid. Fitness stays on ``ind.fitness``; the caller supplies
    the behavior descriptor. ``fitness`` must be single-objective.

    Args:
        centroids: ``(k, dims)`` array of cell centers in descriptor
            space.
    """

    def __init__(self, centroids: Sequence[Sequence[float]] | numpy.ndarray) -> None:
        """See the class docstring."""
        self._centroids = parse_centroids(centroids)
        self._cells: dict[int, Individual] = {}
        self._tree = (
            KDTree(self._centroids) if self._centroids.shape[0] >= KDTREE_MIN_CENTROIDS else None
        )

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[Sequence[float]] | numpy.ndarray,
        k: int,
        *,
        n_iter: int = 20,
    ) -> CvtArchive:
        """Build an archive from k-means centroids of ``samples``.

        Args:
            samples: Behavior descriptors with shape ``(n, dims)``.
            k: Number of centroids.
            n_iter: Independent k-means runs passed to
                :func:`cvt_centroids`.

        Returns:
            An empty archive whose cells are the computed centroids.
        """
        return cls(cvt_centroids(samples, k, n_iter=n_iter))

    @property
    def centroids(self) -> numpy.ndarray:
        """Copy of the ``(k, dims)`` centroid array."""
        return self._centroids.copy()

    @property
    def dimensions(self) -> int:
        """Number of behavior dimensions."""
        return int(self._centroids.shape[1])

    @property
    def stats(self) -> ArchiveStats:
        """Coverage and quality-diversity score of the archive.

        ``num_cells`` is the number of centroids. ``qd_score`` is the
        sum of ``fitness.wvalues[0]`` over elites.
        """
        return make_archive_stats(self._cells.values(), int(self._centroids.shape[0]))

    def nearest_centroid(self, descriptor: Sequence[float]) -> int:
        """Return the index of the centroid nearest to ``descriptor``.

        Archives with fewer than 512 centroids break ties by
        lowest index. Larger archives follow
        ``scipy.spatial.KDTree.query`` order.

        Args:
            descriptor: Continuous behavior coordinates.

        Returns:
            Centroid index in ``0 .. k-1``.

        Raises:
            ValueError: If ``descriptor`` length does not match
                ``dimensions``.
        """
        if len(descriptor) != self.dimensions:
            raise ValueError(
                f"descriptor length {len(descriptor)} does not match {self.dimensions} dimensions"
            )
        return self._centroid_index(descriptor)

    def _centroid_index(self, descriptor: Sequence[float] | numpy.ndarray) -> int:
        """Return the nearest centroid index for ``descriptor``.

        Archives with fewer than :data:`KDTREE_MIN_CENTROIDS` cells use
        :func:`nearest_index` (lowest-index ties). Larger archives use
        ``KDTree.query`` order.

        Args:
            descriptor: Continuous behavior coordinates.

        Returns:
            Centroid index in ``0 .. k-1``.
        """
        if self._tree is None:
            return nearest_index(self._centroids, descriptor)
        return int(self._tree.query(descriptor)[1])

    def add(self, individual: Any, descriptor: Sequence[float] | numpy.ndarray) -> bool:
        """Insert ``individual`` when it improves its Voronoi cell.

        Args:
            individual: Candidate with a valid fitness attribute.
            descriptor: Continuous behavior coordinates.

        Returns:
            True when the archive stores ``individual``.

        Raises:
            ValueError: If ``descriptor`` length does not match
                ``dimensions``, or ``fitness`` is not single-objective.
        """
        if not check_archive_add(individual, descriptor, self.dimensions, "CvtArchive"):
            return False
        cell = self._centroid_index(descriptor)
        return replace_cell_if_better(self._cells, cell, individual)

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
        return elite_at_descriptor(self._cells, descriptor, self.dimensions, self._centroid_index)

    def get(self, index: int) -> Individual | None:
        """Return the elite stored at centroid ``index``.

        Args:
            index: Centroid index in ``0 .. k-1``.

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
        return sample_random_elites(
            list(self._cells.values()),
            n,
            replace=replace,
            empty_message="random_elites from empty CvtArchive",
        )

    def clear(self) -> None:
        """Remove every stored elite."""
        self._cells.clear()

    def __len__(self) -> int:
        """Return the number of filled cells."""
        return len(self._cells)

    def __contains__(self, index: int) -> bool:
        """Return whether centroid ``index`` holds an elite."""
        return index in self._cells

    def __iter__(self) -> Iterator[Individual]:
        """Iterate over stored elites."""
        return iter(self._cells.values())

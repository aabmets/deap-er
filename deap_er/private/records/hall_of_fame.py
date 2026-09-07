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
from bisect import bisect_right
from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
from operator import eq
from typing import TYPE_CHECKING, Any, override

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = ["BaseRecordStorage", "HallOfFame", "ParetoFront"]


def _has_comparable_fitness(individual: Any) -> bool:
    """Return whether ``individual`` has a finite, valid fitness.

    Args:
        individual: Candidate that may lack a fitness attribute.

    Returns:
        True when fitness exists, is valid, and every weighted
        objective is finite.
    """
    if not hasattr(individual, "fitness"):
        return False
    fitness = individual.fitness
    if not fitness.is_valid():
        return False
    return all(math.isfinite(float(value)) for value in fitness.wvalues)


class BaseRecordStorage:
    """Shared storage and ordering for HallOfFame and ParetoFront."""

    def __init__(self) -> None:
        """Create empty item and key lists."""
        self.keys = []
        self.items = []

    def insert(self, individual: Individual) -> None:
        """Insert an individual while preserving sort order; does not enforce maxsize.

        Args:
            individual: Individual to insert. Ignored if it has no
                fitness attribute.
        """
        if hasattr(individual, "fitness"):
            individual = deepcopy(individual)
            i = bisect_right(self.keys, individual.fitness)
            self.items.insert(len(self) - i, individual)
            self.keys.insert(i, individual.fitness)

    def remove(self, index: int) -> None:
        """Remove the individual at ``index``.

        Args:
            index: Position of the individual to remove.

        Raises:
            IndexError: If the hall of fame is empty or ``index`` is
                out of range.
        """
        if not len(self):
            raise IndexError("remove from empty HallOfFame")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("HallOfFame index out of range")
        del self.keys[len(self) - (index + 1)]
        del self.items[index]

    def clear(self) -> None:
        """Remove every stored individual."""
        del self.items[:]
        del self.keys[:]

    def __len__(self) -> int:
        """Return the number of stored individuals."""
        return len(self.items)

    def __getitem__(self, i: int) -> Individual:
        """Return the individual at position ``i``."""
        return self.items[i]

    def __iter__(self) -> Iterator[Individual]:
        """Iterate over individuals from best to worst."""
        return iter(self.items)

    def __reversed__(self) -> Iterator[Individual]:
        """Iterate over individuals from worst to best."""
        return reversed(self.items)

    @override
    def __str__(self) -> str:
        """Return the stored individuals as a string."""
        return str(self.items)


class HallOfFame(BaseRecordStorage):
    """Archive of the best individuals seen during evolution.

    Members stay sorted by fitness so the first item is the best
    individual seen so far, according to the fitness weights.

    Args:
        maxsize: Maximum number of individuals to keep.
        similar: Equality test used to skip duplicates. Defaults to
            ``operator.eq``.
    """

    def __init__(self, maxsize: int, similar: Callable[..., Any] = eq) -> None:
        """See the class docstring."""
        self.maxsize = maxsize
        self.similar = similar
        super().__init__()

    def _similar_index(self, individual: Any) -> int | None:
        """Return the index of a stored member similar to ``individual``.

        Args:
            individual: Candidate to compare against stored members.

        Returns:
            Index of the first similar member, or None.
        """
        return next(
            (i for i, member in enumerate(self) if self.similar(individual, member)),
            None,
        )

    def _update_one(self, individual: Any) -> None:
        """Insert or replace ``individual`` if it belongs in the archive.

        Args:
            individual: Candidate with or without a fitness attribute.
        """
        if not _has_comparable_fitness(individual):
            return
        if len(self) == 0:
            self.insert(individual)
            return
        similar_index = self._similar_index(individual)
        if similar_index is not None:
            if individual.fitness > self[similar_index].fitness:
                self.remove(similar_index)
                self.insert(individual)
            return
        if individual.fitness > self[-1].fitness or len(self) < self.maxsize:
            if len(self) >= self.maxsize:
                self.remove(-1)
            self.insert(individual)

    def update(self, population: Sequence[Any]) -> None:
        """Update the archive from ``population``.

        Better individuals replace the worst members. The archive stays
        at most ``maxsize`` and skips individuals already present
        according to ``similar``. Individuals without a comparable
        fitness (missing, invalid, or non-finite) are ignored.

        Args:
            population: Individuals that may have a fitness attribute.
        """
        if self.maxsize == 0:
            return
        for ind in population:
            self._update_one(ind)


class ParetoFront(BaseRecordStorage):
    """Archive of every non-dominated individual seen during evolution.

    The front is unbounded: every unique non-dominated individual is kept.

    Args:
        similar: Equality test used to skip duplicates. Defaults to
            ``operator.eq``.
    """

    def __init__(self, similar: Callable[..., Any] = eq) -> None:
        """See the class docstring."""
        self.similar = similar
        super().__init__()

    def _front_verdict(self, individual: Any) -> tuple[bool, bool, list[int]]:
        """Compare ``individual`` to the current front.

        Args:
            individual: Candidate that has a fitness attribute.

        Returns:
            Whether the front dominates it, whether a twin exists, and
            indexes of members it dominates.
        """
        is_dominated = False
        dominates_one = False
        has_twin = False
        to_remove = []
        for i, hof_member in enumerate(self):
            if not dominates_one and hof_member.fitness.dominates(individual.fitness):
                is_dominated = True
                break
            if individual.fitness.dominates(hof_member.fitness):
                dominates_one = True
                to_remove.append(i)
            elif individual.fitness == hof_member.fitness and self.similar(individual, hof_member):
                has_twin = True
                break
        return is_dominated, has_twin, to_remove

    def update(self, population: Sequence[Any]) -> None:
        """Add non-dominated individuals from ``population``.

        Members dominated by a new individual are removed. Similar
        individuals with equal fitness are not added again.
        Individuals without a comparable fitness (missing, invalid,
        or non-finite) are ignored.

        Args:
            population: Individuals that may have a fitness attribute.
        """
        for ind in population:
            if not _has_comparable_fitness(ind):
                continue
            is_dominated, has_twin, to_remove = self._front_verdict(ind)
            for i in reversed(to_remove):
                self.remove(i)
            if not is_dominated and not has_twin:
                self.insert(ind)

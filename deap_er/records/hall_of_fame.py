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
from bisect import bisect_right
from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
from operator import eq
from typing import Any, override

from deap_er.base.dtypes import *

__all__ = ["HallOfFame", "ParetoFront"]


class _BaseClass:
    """Shared storage and ordering for HallOfFame and ParetoFront."""

    def __init__(self) -> None:
        """Create empty item and key lists."""
        self.keys = list()
        self.items = list()

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
        """
        del self.keys[len(self) - (index % len(self) + 1)]
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


class HallOfFame(_BaseClass):
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

    def update(self, population: Sequence[Any]) -> None:
        """Update the archive from ``population``.

        Better individuals replace the worst members. The archive stays
        at most ``maxsize`` and skips individuals already present
        according to ``similar``.

        Args:
            population: Individuals with a fitness attribute.
        """
        for ind in population:
            if len(self) == 0 and self.maxsize != 0:
                self.insert(population[0])
                continue
            if ind.fitness > self[-1].fitness or len(self) < self.maxsize:
                for hof_member in self:
                    if self.similar(ind, hof_member):
                        break
                else:
                    if len(self) >= self.maxsize:
                        self.remove(-1)
                    self.insert(ind)


class ParetoFront(_BaseClass):
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

    def update(self, population: Sequence[Any]) -> None:
        """Add non-dominated individuals from ``population``.

        Members dominated by a new individual are removed. Similar
        individuals with equal fitness are not added again.

        Args:
            population: Individuals with a fitness attribute.
        """
        for ind in population:
            is_dominated = False
            dominates_one = False
            has_twin = False
            to_remove = list()
            for i, hof_member in enumerate(self):
                if not dominates_one and hof_member.fitness.dominates(ind.fitness):
                    is_dominated = True
                    break
                elif ind.fitness.dominates(hof_member.fitness):
                    dominates_one = True
                    to_remove.append(i)
                elif ind.fitness == hof_member.fitness and self.similar(ind, hof_member):
                    has_twin = True
                    break

            for i in reversed(to_remove):
                self.remove(i)
            if not is_dominated and not has_twin:
                self.insert(ind)

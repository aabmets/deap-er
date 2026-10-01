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

from deap_er.private.fitness import has_comparable_fitness

from .hof_json import hall_of_fame_from_json, hall_of_fame_to_json

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = ["BaseRecordStorage", "HallOfFame"]


class BaseRecordStorage:
    """Shared storage and ordering for HallOfFame and ParetoFront."""

    def __init__(self) -> None:
        """Create empty item and key lists."""
        self.keys = []
        self.items = []

    def _rank(self, individual: Any) -> Any:
        """Return the sort key of ``individual``; larger is better."""
        return individual.fitness

    def _is_rankable(self, individual: Any) -> bool:
        """Return whether ``individual`` can be stored and ordered."""
        return has_comparable_fitness(individual)

    def insert(self, individual: Individual) -> None:
        """Insert an individual while preserving sort order; does not enforce maxsize.

        Args:
            individual: Individual to insert. Ignored if fitness is
                missing, invalid, or non-finite, or if a ``key`` is set
                and its value is non-finite.
        """
        if self._is_rankable(individual):
            individual = deepcopy(individual)
            rank = self._rank(individual)
            i = bisect_right(self.keys, rank)
            self.items.insert(len(self) - i, individual)
            self.keys.insert(i, rank)

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
    individual seen so far, according to the fitness weights. With
    ``key``, members are ranked by ``key(individual)`` instead, which
    lets a multi-objective individual be ranked by one scalar.

    Args:
        maxsize: Maximum number of individuals to keep.
        similar: Equality test used to skip duplicates. Defaults to
            ``operator.eq``.
        key: Optional scalariser mapping an individual to a float where
            **larger is better** (negate a cost to minimise it). The
            individual still needs a comparable fitness; individuals
            whose key value is non-finite are ignored. Defaults to
            None, which ranks by the fitness weights. A lambda key needs
            dill (as used by ``Checkpoint``) to be pickled.

    Raises:
        ValueError: If ``maxsize`` is negative.
    """

    def __init__(
        self,
        maxsize: int,
        similar: Callable[..., Any] = eq,
        key: Callable[[Any], float] | None = None,
    ) -> None:
        """See the class docstring."""
        if maxsize < 0:
            raise ValueError("maxsize must be non-negative")
        self.maxsize = maxsize
        self.similar = similar
        self.key = key
        super().__init__()

    @override
    def _rank(self, individual: Any) -> Any:
        if self.key is None:
            return individual.fitness
        return float(self.key(individual))

    @override
    def _is_rankable(self, individual: Any) -> bool:
        if not has_comparable_fitness(individual):
            return False
        return self.key is None or math.isfinite(self._rank(individual))

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
        if not self._is_rankable(individual):
            return
        if len(self) == 0:
            self.insert(individual)
            return
        rank = self._rank(individual)
        similar_index = self._similar_index(individual)
        if similar_index is not None:
            if rank > self._rank(self[similar_index]):
                self.remove(similar_index)
                self.insert(individual)
            return
        if rank > self._rank(self[-1]) or len(self) < self.maxsize:
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

    def to_json(self) -> str:
        """Serialize ``maxsize`` and archive members to JSON."""
        return hall_of_fame_to_json(self)

    @classmethod
    def from_json(
        cls,
        text: str,
        ind_cls: type[Any] | None = None,
        key: Callable[[Any], float] | None = None,
    ) -> HallOfFame:
        """Rebuild a hall of fame from :meth:`to_json` output.

        ``key`` is not serialized; pass it again to restore a keyed
        archive.
        """
        return hall_of_fame_from_json(text, ind_cls, cls, key)

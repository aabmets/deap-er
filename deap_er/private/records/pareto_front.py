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

from collections.abc import Callable, Sequence
from operator import eq
from typing import Any

from deap_er.private.fitness import has_comparable_fitness

from .hall_of_fame import BaseRecordStorage

__all__: list[str] = ["ParetoFront"]


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
            if not has_comparable_fitness(ind):
                continue
            is_dominated, has_twin, to_remove = self._front_verdict(ind)
            for i in reversed(to_remove):
                self.remove(i)
            if not is_dominated and not has_twin:
                self.insert(ind)

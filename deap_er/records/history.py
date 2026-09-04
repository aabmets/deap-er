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
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from deap_er.base.dtypes import Individual

__all__ = ["History"]


class History:
    """Genealogy of individuals produced during evolution.

    Call ``update`` on the initial population and after each variation,
    or wrap variation operators with ``decorator``.
    """

    def __init__(self) -> None:
        """Create an empty genealogy."""
        self.genealogy_index = 0
        self.genealogy_history = {}
        self.genealogy_tree = {}

    @property
    def decorator(self) -> Callable[..., Any]:
        """Decorator that records a variation operator's returned individuals."""

        def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                individuals = func(*args, **kwargs)
                self.update(individuals)
                return individuals

            return wrapped

        return wrapper

    def update(self, individuals: list[Individual]) -> None:
        """Record ``individuals`` in the genealogy.

        Call this on the initial population and after each variation.
        Individuals that already have ``history_index`` become the
        parents of the newly recorded entries; otherwise the entries
        are roots.

        Args:
            individuals: Individuals to add to the genealogy.
        """
        try:
            parent_indices = tuple(ind.history_index for ind in individuals)
        except AttributeError:
            parent_indices = ()

        for ind in individuals:
            self.genealogy_index += 1
            ind.history_index = self.genealogy_index
            self.genealogy_history[self.genealogy_index] = deepcopy(ind)
            self.genealogy_tree[self.genealogy_index] = parent_indices

    def get_genealogy(
        self, individual: Individual, max_depth: float = float("inf")
    ) -> dict[int, Any]:
        """Return the ancestor graph of an individual.

        The individual must have a ``history_index`` set by ``update``.
        The graph includes parents up to ``max_depth`` variation steps.
        The default ``max_depth`` walks back to the start of the
        evolution.

        Args:
            individual: Individual at the root of the genealogy tree.
            max_depth: Maximum number of variation steps to walk.

        Returns:
            Mapping of individual index to a tuple of parent indices.

        Raises:
            AttributeError: If the individual has no ``history_index``.
        """

        def _recursive(index: int, depth: int) -> None:
            if index not in self.genealogy_tree:
                return
            depth += 1
            if depth > max_depth:
                return
            parent_indices = self.genealogy_tree[index]
            gtree[index] = parent_indices
            for ind in parent_indices:
                if ind not in visited:
                    _recursive(ind, depth)
                visited.add(ind)

        if hasattr(individual, "history_index"):
            visited = set()
            gtree = {}
            _recursive(individual.history_index, 0)
            return gtree
        else:
            raise AttributeError("The individual must have the 'history_index' attribute.")

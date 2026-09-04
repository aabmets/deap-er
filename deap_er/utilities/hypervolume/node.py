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

from typing import Callable
from operator import gt, ge, le, lt, eq, ne


class Node:
    """A shared node in a Fonseca hypervolume multi-list.

    Each node belongs to one list per dimension and stores the cargo
    used by the recursive hypervolume indicator.

    Args:
        dimensions: Number of list dimensions.
        cargo: Point coordinates stored on the node. Optional.
    """

    def __init__(self, dimensions: int, cargo: tuple = None):
        """See the class docstring."""
        self.cargo = cargo
        self.next = [None] * dimensions
        self.prev = [None] * dimensions
        self.ignore = 0
        self.area = [0.0] * dimensions
        self.volume = [0.0] * dimensions

    def compare(self, other: Node, op: Callable) -> bool:
        """Return whether ``op`` holds for every cargo coordinate pair.

        Returns False if either node has no cargo.

        Args:
            other: Node to compare against.
            op: Binary comparison applied to each coordinate pair.

        Returns:
            True if ``op`` holds for all coordinates.
        """
        if self.cargo is None or other.cargo is None:
            return False
        zipper = zip(self.cargo, other.cargo)
        true = [op(a, b) for a, b in zipper]
        return all(true)

    def __gt__(self, other: Node) -> bool:
        """Return whether every cargo coordinate is greater than ``other``'s."""
        return self.compare(other, gt)

    def __ge__(self, other: Node) -> bool:
        """Return whether every cargo coordinate is at least ``other``'s."""
        return self.compare(other, ge)

    def __le__(self, other: Node) -> bool:
        """Return whether every cargo coordinate is at most ``other``'s."""
        return self.compare(other, le)

    def __lt__(self, other: Node) -> bool:
        """Return whether every cargo coordinate is less than ``other``'s."""
        return self.compare(other, lt)

    def __eq__(self, other: Node) -> bool:
        """Return whether the cargo coordinates compare equal."""
        return self.compare(other, eq)

    def __ne__(self, other: Node) -> bool:
        """Return whether the cargo coordinates compare unequal."""
        return self.compare(other, ne)

    def __str__(self) -> str:
        """Return the cargo as a string."""
        return str(self.cargo)

    def __hash__(self) -> int:
        """Return the hash of the cargo."""
        return hash(self.cargo)

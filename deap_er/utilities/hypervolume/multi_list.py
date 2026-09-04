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
from collections.abc import Iterable, MutableSequence

from .node import Node


class MultiList:
    """Linked lists that share nodes across dimensions.

    Used by the Fonseca hypervolume indicator. Every node has one
    predecessor and one successor in each dimension.

    Args:
        dimensions: Number of dimensions in the multi-list.
    """

    def __init__(self, dimensions: int) -> None:
        """See the class docstring."""
        self.dimensions = dimensions
        self.sentinel = Node(dimensions)
        self.sentinel.next = [self.sentinel] * dimensions
        self.sentinel.prev = [self.sentinel] * dimensions

    def __str__(self) -> str:
        """Return a per-dimension listing of node cargo."""
        strings = list()
        for i in range(self.dimensions):
            current_list = list()
            node = self.sentinel.next[i]
            while node != self.sentinel:
                current_list.append(str(node))
                node = node.next[i]
            strings.append(str(current_list))
        _repr = ""
        for string in strings:
            _repr += string + "\n"
        return _repr

    def __len__(self) -> int:
        """Return the number of dimensions."""
        return self.dimensions

    def get_length(self, index: int) -> int:
        """Return the number of nodes in the list at dimension ``index``.

        Args:
            index: Dimension of the list to measure.

        Returns:
            Node count, excluding the sentinel.
        """
        length = 0
        node = self.sentinel.next[index]
        while node != self.sentinel:
            node = node.next[index]
            length += 1
        return length

    def append(self, node: Node, index: int) -> None:
        """Append ``node`` to the list at dimension ``index``.

        Args:
            node: Node to append.
            index: Dimension of the list to extend.
        """
        penultimate = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = penultimate
        self.sentinel.prev[index] = node
        penultimate.next[index] = node

    def extend(self, nodes: Iterable[Node], index: int) -> None:
        """Append each node in ``nodes`` to the list at dimension ``index``.

        Args:
            nodes: Nodes to append, in order.
            index: Dimension of the list to extend.
        """
        for node in nodes:
            penultimate = self.sentinel.prev[index]
            node.next[index] = self.sentinel
            node.prev[index] = penultimate
            self.sentinel.prev[index] = node
            penultimate.next[index] = node

    @staticmethod
    def remove(node: Node, index: int, bounds: MutableSequence) -> Node:
        """Unlink ``node`` from lists in dimensions below ``index``.

        Tightens ``bounds`` when the removed cargo is smaller on a
        dimension.

        Args:
            node: Node to unlink.
            index: Exclusive upper bound on dimensions to unlink.
            bounds: Per-dimension bounds updated in place.

        Returns:
            The unlinked node.
        """
        for i in range(index):
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
        return node

    @staticmethod
    def reinsert(node: Node, index: int, bounds: MutableSequence) -> None:
        """Restore ``node`` into lists in dimensions below ``index``.

        Tightens ``bounds`` when the restored cargo is smaller on a
        dimension.

        Args:
            node: Node to restore.
            index: Exclusive upper bound on dimensions to relink.
            bounds: Per-dimension bounds updated in place.
        """
        for i in range(index):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]

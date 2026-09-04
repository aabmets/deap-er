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
from collections.abc import Iterable, Iterator
from itertools import product
from typing import Any

__all__ = ["SortingNetwork"]


class SortingNetwork:
    """A network of wires and comparators that sorts a sequence.

    Wires run from left to right and carry one value each. A
    comparator connects two wires and swaps their values when the
    upper wire is greater than the lower wire.

    Args:
        dimension: Number of wires in the network.
        connectors: Optional list of wire pairs connected by a
            comparator.
    """

    def __init__(self, dimension: int, connectors: list[tuple[int, int]] | None = None) -> None:
        """See the class docstring."""
        self.dimension = dimension
        self.data: list[list[tuple[int, int]]] = []
        if connectors:
            for wire1, wire2 in connectors:
                self.add_connector(wire1, wire2)
        super().__init__()

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        """Iterate over comparator levels."""
        return iter(self.data)

    def __contains__(self, item: object) -> bool:
        """Return whether ``item`` is a stored level."""
        return item in self.data

    def __getitem__(self, key: int) -> list[tuple[int, int]]:
        """Return the comparator level at ``key``."""
        return self.data[key]

    def __setitem__(self, key: int, value: list[tuple[int, int]]) -> None:
        """Replace the comparator level at ``key``."""
        self.data[key] = value

    def __delitem__(self, key: int) -> None:
        """Delete the comparator level at ``key``."""
        del self.data[key]

    def __len__(self) -> int:
        """Return the number of comparator levels."""
        return len(self.data)

    @property
    def depth(self) -> int:
        """Returns the depth of the network."""
        return len(self.data) if self.data else 0

    @property
    def length(self) -> int:
        """Returns the length of the network."""
        return sum(len(level) for level in self.data)

    @staticmethod
    def check_conflict(level: list[tuple[int, int]], wire1: int, wire2: int) -> bool:
        """Return whether the wires conflict on the given level.

        Args:
            level: Comparators already present on the level.
            wire1: Index of the first wire.
            wire2: Index of the second wire.

        Returns:
            True if the wires conflict, False otherwise.
        """
        return any(wires[1] >= wire1 and wires[0] <= wire2 for wires in level)

    def add_connector(self, wire1: int, wire2: int) -> None:
        """Add a comparator between the two wires.

        Same-index wires are ignored.

        Args:
            wire1: Index of the first wire.
            wire2: Index of the second wire.
        """
        if wire1 == wire2:
            return

        if wire1 > wire2:
            wire1, wire2 = wire2, wire1

        index = 0
        for level in reversed(self.data):
            if self.check_conflict(level, wire1, wire2):
                break
            index -= 1

        cnx = (wire1, wire2)
        if index == 0:
            self.data.append([cnx])
        else:
            self.data[index].append(cnx)

    def sort(self, values: list[Any]) -> None:
        """Sort ``values`` in place using this network.

        Args:
            values: Sequence to sort. Must have at least ``dimension``
                elements.
        """
        for level in self.data:
            for wire1, wire2 in level:
                if values[wire1] > values[wire2]:
                    values[wire1], values[wire2] = values[wire2], values[wire1]

    def evaluate(self, cases: Iterable[Iterable[Any]] | None = None) -> int:
        """Count how many ``cases`` the network fails to sort.

        When ``cases`` is omitted, every binary sequence of length
        ``dimension`` is tested.

        Args:
            cases: Sequences to sort and check. Optional.

        Returns:
            The number of incorrectly sorted cases.
        """
        if cases is None:
            cases = product((0, 1), repeat=self.dimension)

        errors = 0
        ordered = []
        for i in range(self.dimension + 1):
            result = [0] * (self.dimension - i) + [1] * i
            ordered.append(result)
        for sequence in cases:
            sequence = list(sequence)
            self.sort(sequence)
            idx = sum(sequence)
            errors += int(sequence != ordered[idx])
        return errors

    def draw(self) -> str:
        """Return an ASCII diagram of the network.

        Returns:
            A schematic of the wires and comparators.
        """
        str_wires = [["-"] * 7 * self.depth]
        str_wires[0][0] = "0"
        str_wires[0][1] = " o"
        str_spaces = []

        for i in range(1, self.dimension):
            str_wires.append(["-"] * 7 * self.depth)
            str_spaces.append([" "] * 7 * self.depth)
            str_wires[i][0] = str(i)
            str_wires[i][1] = " o"

        for index, level in enumerate(self.data):
            for wire1, wire2 in level:
                str_wires[wire1][(index + 1) * 6] = "x"
                str_wires[wire2][(index + 1) * 6] = "x"
                for i in range(wire1, wire2):
                    str_spaces[i][(index + 1) * 6 + 1] = "|"
                for i in range(wire1 + 1, wire2):
                    str_wires[i][(index + 1) * 6] = "|"

        network_draw = "".join(str_wires[0])

        for line, space in zip(str_wires[1:], str_spaces, strict=False):
            network_draw += "\n"
            network_draw += "".join(space)
            network_draw += "\n"
            network_draw += "".join(line)

        return network_draw

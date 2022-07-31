# ====================================================================================== #
#                                                                                        #
#   MIT License                                                                          #
#                                                                                        #
#   Copyright (c) 2022 - Mattias Aabmets, The DEAP Team and Other Contributors           #
#                                                                                        #
#   Permission is hereby granted, free of charge, to any person obtaining a copy         #
#   of this software and associated documentation files (the "Software"), to deal        #
#   in the Software without restriction, including without limitation the rights         #
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell            #
#   copies of the Software, and to permit persons to whom the Software is                #
#   furnished to do so, subject to the following conditions:                             #
#                                                                                        #
#   The above copyright notice and this permission notice shall be included in all       #
#   copies or substantial portions of the Software.                                      #
#                                                                                        #
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR           #
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,             #
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE          #
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER               #
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,        #
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE        #
#   SOFTWARE.                                                                            #
#                                                                                        #
# ====================================================================================== #
from itertools import product


__all__ = ['SortingNetwork']


# ====================================================================================== #
class SortingNetwork:
    """
    A sorting network is an abstract mathematical model of a network of wires
    and comparator modules that is used to sort a sequence of numbers. The wires
    are thought of as running from left to right, carrying values (one per wire)
    that traverse the network all at the same time. Each comparator connects two
    wires. When a pair of values, traveling through a pair of wires, encounter a
    comparator, the comparator swaps the values if and only if the top wire's
    value is greater or equal to the bottom wire's value.

    :param: dimension: The number of wires in the network.
    :param: connectors: A list of pairs of wires
        that are connected by a comparator.
    """
    # -------------------------------------------------------- #
    def __init__(self, dimension: int, connectors: list = None):
        self.dimension = dimension
        self.data = list()
        if connectors:
            for wire1, wire2 in connectors:
                self.add_connector(wire1, wire2)
        super().__init__()

    # -------------------------------------------------------- #
    @property
    def depth(self) -> int:
        """Returns the depth of the network."""
        return len(self.data) if self.data else 0

    # -------------------------------------------------------- #
    @property
    def length(self):
        """Returns the length of the network."""
        return sum(len(level) for level in self.data)

    # -------------------------------------------------------- #
    @staticmethod
    def check_conflict(level, wire1, wire2) -> bool:
        """Checks if the given wires are in conflict with each other."""
        for wires in level:
            if wires[1] >= wire1 and wires[0] <= wire2:
                return True
        return False

    # -------------------------------------------------------- #
    def add_connector(self, wire1, wire2) -> None:
        """Adds a connector to the network."""
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

    # -------------------------------------------------------- #
    def sort(self, values: list) -> None:
        """Sorts the given values using the network."""
        for level in self.data:
            for wire1, wire2 in level:
                if values[wire1] > values[wire2]:
                    values[wire1], values[wire2] = values[wire2], values[wire1]

    # -------------------------------------------------------- #
    def assess(self, cases: list = None) -> int:
        """Assesses the network for the given cases."""
        if cases is None:
            cases = product((0, 1), repeat=self.dimension)

        misses = 0
        ordered = []
        for i in range(self.dimension + 1):
            result = [0] * (self.dimension - i) + [1] * i
            ordered.append(result)
        for sequence in cases:
            sequence = list(sequence)
            self.sort(sequence)
            misses += (sequence != ordered[sum(sequence)])
        return misses

    # -------------------------------------------------------- #
    def draw(self) -> str:
        """Draws the network."""
        str_wires = [["-"] * 7 * self.depth]
        str_wires[0][0] = "0"
        str_wires[0][1] = " o"
        str_spaces = []

        for i in range(1, self.dimension):
            str_wires.append(["-"]*7 * self.depth)
            str_spaces.append([" "]*7 * self.depth)
            str_wires[i][0] = str(i)
            str_wires[i][1] = " o"

        for index, level in enumerate(self.data):
            for wire1, wire2 in level:
                str_wires[wire1][(index+1)*6] = "x"
                str_wires[wire2][(index+1)*6] = "x"
                for i in range(wire1, wire2):
                    str_spaces[i][(index+1)*6+1] = "|"
                for i in range(wire1+1, wire2):
                    str_wires[i][(index+1)*6] = "|"

        network_draw = "".join(str_wires[0])

        for line, space in zip(str_wires[1:], str_spaces):
            network_draw += "\n"
            network_draw += "".join(space)
            network_draw += "\n"
            network_draw += "".join(line)

        return network_draw

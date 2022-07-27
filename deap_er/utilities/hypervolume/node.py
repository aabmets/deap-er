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
from __future__ import annotations
from typing import Callable
from operator import gt, ge, le, lt, eq, ne


# ====================================================================================== #
class Node:
    def __init__(self, dimensions: int, cargo: tuple = None):
        self.cargo = cargo
        self.next = [None] * dimensions
        self.prev = [None] * dimensions
        self.ignore = 0
        self.area = [0.0] * dimensions
        self.volume = [0.0] * dimensions

    # -------------------------------------------------------- #
    def compare(self, other: Node, op: Callable) -> bool:
        if self.cargo is None or other.cargo is None:
            return False
        zipper = zip(self.cargo, other.cargo)
        true = [op(a, b) for a, b in zipper]
        return all(true)

    # -------------------------------------------------------- #
    def __gt__(self, other: Node) -> bool:
        return self.compare(other, gt)

    def __ge__(self, other: Node) -> bool:
        return self.compare(other, ge)

    def __le__(self, other: Node) -> bool:
        return self.compare(other, le)

    def __lt__(self, other: Node) -> bool:
        return self.compare(other, lt)

    def __eq__(self, other: Node) -> bool:
        return self.compare(other, eq)

    def __ne__(self, other: Node) -> bool:
        return self.compare(other, ne)

    # -------------------------------------------------------- #
    def __str__(self) -> str:
        return str(self.cargo)

    # -------------------------------------------------------- #
    def __hash__(self) -> int:
        return hash(self.cargo)

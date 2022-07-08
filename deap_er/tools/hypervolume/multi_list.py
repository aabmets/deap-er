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
from deap_er._deprecated import deprecated
from collections.abc import Iterable, MutableSequence
from .node import Node


# ====================================================================================== #
class MultiList:
    """
    A special data structure needed by the Fonseca HyperVolume indicator.
    It consists of several doubly linked lists that share common nodes.
    Every node has multiple predecessors and successors, one in every list.
    """
    # -------------------------------------------------------------------------------------- #
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.sentinel = Node(dimensions)
        self.sentinel.next = [self.sentinel] * dimensions
        self.sentinel.prev = [self.sentinel] * dimensions

    # -------------------------------------------------------------------------------------- #
    def __str__(self) -> str:
        strings = list()
        for i in range(self.dimensions):
            current_list = list()
            node = self.sentinel.next[i]
            while node != self.sentinel:
                current_list.append(str(node))
                node = node.next[i]
            strings.append(str(current_list))
        _repr = str()
        for string in strings:
            _repr += string + "\n"
        return _repr

    # -------------------------------------------------------------------------------------- #
    def __len__(self):
        return self.dimensions

    # -------------------------------------------------------------------------------------- #
    def get_length(self, index: int) -> int:
        length = 0
        node = self.sentinel.next[index]
        while node != self.sentinel:
            node = node.next[index]
            length += 1
        return length

    # -------------------------------------------------------------------------------------- #
    def append(self, node: Node, index: int) -> None:
        penultimate = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = penultimate
        self.sentinel.prev[index] = node
        penultimate.next[index] = node

    # -------------------------------------------------------------------------------------- #
    def extend(self, nodes: Iterable[Node], index: int) -> None:
        for node in nodes:
            penultimate = self.sentinel.prev[index]
            node.next[index] = self.sentinel
            node.prev[index] = penultimate
            self.sentinel.prev[index] = node
            penultimate.next[index] = node

    # -------------------------------------------------------------------------------------- #
    @staticmethod
    def remove(node: Node, index: int, bounds: MutableSequence) -> Node:
        for i in range(index):
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
        return node

    # -------------------------------------------------------------------------------------- #
    @staticmethod
    def reinsert(node: Node, index: int, bounds: MutableSequence) -> None:
        for i in range(index):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]

    # -------------------------------------------------------------------------------------- #
    getLength = deprecated('getLength', get_length)

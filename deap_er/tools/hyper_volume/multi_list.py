# ====================================================================================== #
#                                                                                        #
#   MIT License                                                                          #
#                                                                                        #
#   Copyright (c) 2022 The Original DEAP Team, Mattias Aabmets and Contributors          #
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
from node import Node


# ====================================================================================== #
class MultiList:
    """
    A special data structure needed by FonsecaHyperVolume.
    It consists of several doubly linked lists that share common nodes.
    Every node has multiple predecessors and successors, one in every list.
    """

    # -------------------------------------------------------------------------------------- #
    def __init__(self, dimensions):
        """
        Constructor.
        Builds 'numberLists' doubly linked lists.
        """
        self.numberLists = dimensions
        self.sentinel = Node(dimensions)
        self.sentinel.next = [self.sentinel] * dimensions
        self.sentinel.prev = [self.sentinel] * dimensions

    # -------------------------------------------------------------------------------------- #
    def __str__(self):
        """
        Returns the string representation of the internal numberLists variable.
        """
        strings = []
        for i in range(self.numberLists):
            current_list = []
            node = self.sentinel.next[i]
            while node != self.sentinel:
                current_list.append(str(node))
                node = node.next[i]
            strings.append(str(current_list))
        string_repr = ""
        for string in strings:
            string_repr += string + "\n"
        return string_repr

    # -------------------------------------------------------------------------------------- #
    def __len__(self):
        """
        Returns the number of lists that are included in this MultiList.
        """
        return self.numberLists

    # -------------------------------------------------------------------------------------- #
    def getLength(self, i):
        """
        Returns the length of the i-th list.
        """
        length = 0
        node = self.sentinel.next[i]
        while node != self.sentinel:
            length += 1
            node = node.next[i]
        return length

    # -------------------------------------------------------------------------------------- #
    def append(self, node, index):
        """
        Appends a node to the end of the list at the given index.
        """
        last_but_one = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = last_but_one
        self.sentinel.prev[index] = node
        last_but_one.next[index] = node

    # -------------------------------------------------------------------------------------- #
    def extend(self, nodes, index):
        """
        Extends the list at the given index with the nodes.
        """
        sentinel = self.sentinel
        for node in nodes:
            last_but_one = sentinel.prev[index]
            node.next[index] = sentinel
            node.prev[index] = last_but_one
            sentinel.prev[index] = node
            last_but_one.next[index] = node

    # -------------------------------------------------------------------------------------- #
    def remove(self, node, index, bounds):
        """
        Removes and returns 'node' from all lists in [0, 'index'[.
        """
        for i in range(index):
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
        return node

    # -------------------------------------------------------------------------------------- #
    def reinsert(self, node, index, bounds):
        """
        Inserts 'node' at the position it had in all lists in [0, 'index'[
        before it was removed. This method assumes that the next and previous
        nodes of the node that is reinserted are in the list.
        """
        for i in range(index):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]

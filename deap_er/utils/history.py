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
from deap_er.deprecated import deprecated
from copy import deepcopy


__all__ = ['History']


# ====================================================================================== #
class History:
    def __init__(self):
        self.genealogy_index = int()
        self.genealogy_history = dict()
        self.genealogy_tree = dict()

    # -------------------------------------------------------------------------------------- #
    @property
    def decorator(self):
        def wrapper(func):
            def wrapped(*args, **kwargs):
                individuals = func(*args, **kwargs)
                self.update(individuals)
                return individuals
            return wrapped
        return wrapper

    # -------------------------------------------------------------------------------------- #
    def update(self, individuals):
        try:
            parent_indices = tuple(ind.history_index for ind in individuals)
        except AttributeError:
            parent_indices = tuple()

        for ind in individuals:
            self.genealogy_index += 1
            ind.history_index = self.genealogy_index
            self.genealogy_history[self.genealogy_index] = deepcopy(ind)
            self.genealogy_tree[self.genealogy_index] = parent_indices

    # -------------------------------------------------------------------------------------- #
    def get_genealogy(self, individual, max_depth=float("inf")):
        def _recursive(index, depth):
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

        visited = set()
        gtree = dict()

        _recursive(individual.history_index, 0)
        return gtree

    # -------------------------------------------------------------------------------------- #
    getGenealogy = deprecated('getGenealogy', get_genealogy)

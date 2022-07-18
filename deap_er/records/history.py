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
from typing import Callable
from copy import deepcopy


__all__ = ['History']


# ====================================================================================== #
class History:
    """
    The *History* class helps with building a genealogy
    of the individuals produced in the evolution.
    """
    # -------------------------------------------------------- #
    def __init__(self):
        self.genealogy_index = int()
        self.genealogy_history = dict()
        self.genealogy_tree = dict()

    # -------------------------------------------------------- #
    @property
    def decorator(self) -> Callable:
        """
        Property that returns an appropriate decorator
        to enhance the operators of the toolbox.

        Returns:
            A decorator that adds the genealogy history to the individuals.
        """
        def wrapper(func):
            def wrapped(*args, **kwargs):
                individuals = func(*args, **kwargs)
                self.update(individuals)
                return individuals
            return wrapped
        return wrapper

    # -------------------------------------------------------- #
    def update(self, individuals: list) -> None:
        """
        Update the genealogy history with the given individuals.
        This method should be called with the initial population
        to initialize the history and also after each variation.

        Parameters:
            individuals: The individuals to update the genealogy history with.
        Returns:
            None
        """
        try:
            parent_indices = tuple(ind.history_index for ind in individuals)
        except AttributeError:
            parent_indices = tuple()

        for ind in individuals:
            self.genealogy_index += 1
            ind.history_index = self.genealogy_index
            self.genealogy_history[self.genealogy_index] = deepcopy(ind)
            self.genealogy_tree[self.genealogy_index] = parent_indices

    # -------------------------------------------------------- #
    def get_genealogy(self, individual, max_depth=float("inf")) -> dict:
        """
        Get the genealogy of the given individual. The returned graph contains
        the parents up to *max_depth* variations before this individual. The
        default value of *max_depth* is up to the beginning of the evolution.

        Parameters:
            individual: The individual at the root of the genealogy tree.
            max_depth: The maximum depth of the genealogy tree.
        Returns: A dictionary where each key is an individual index and the
            values are tuples corresponding to the index of the parents.
        """
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

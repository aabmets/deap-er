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
from bisect import bisect_right
from copy import deepcopy
from operator import eq


__all__ = ['HallOfFame', 'ParetoFront']


# ====================================================================================== #
class _BaseClass:

    def __init__(self):
        self.keys = list()
        self.items = list()

    # -------------------------------------------------------------------------------------- #
    def insert(self, item):
        item = deepcopy(item)
        i = bisect_right(self.keys, item.fitness)
        self.items.insert(len(self) - i, item)
        self.keys.insert(i, item.fitness)

    # -------------------------------------------------------------------------------------- #
    def remove(self, index):
        del self.keys[len(self) - (index % len(self) + 1)]
        del self.items[index]

    # -------------------------------------------------------------------------------------- #
    def clear(self):
        del self.items[:]
        del self.keys[:]

    # -------------------------------------------------------------------------------------- #
    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __iter__(self):
        return iter(self.items)

    def __reversed__(self):
        return reversed(self.items)

    def __str__(self):
        return str(self.items)


# ====================================================================================== #
class HallOfFame(_BaseClass):

    def __init__(self, maxsize: int, similar: Callable = eq):
        self.maxsize = maxsize
        self.similar = similar
        super().__init__()

    # -------------------------------------------------------------------------------------- #
    def update(self, population):
        for ind in population:
            if len(self) == 0 and self.maxsize != 0:
                self.insert(population[0])
                continue
            if ind.fitness > self[-1].fitness or len(self) < self.maxsize:
                for hof_member in self:
                    if self.similar(ind, hof_member):
                        break
                else:
                    if len(self) >= self.maxsize:
                        self.remove(-1)
                    self.insert(ind)


# ====================================================================================== #
class ParetoFront(_BaseClass):

    def __init__(self, similar: Callable = eq):
        self.similar = similar
        super().__init__()

    # -------------------------------------------------------------------------------------- #
    def update(self, population):
        for ind in population:
            is_dominated = False
            dominates_one = False
            has_twin = False
            to_remove = list()
            for i, hof_member in enumerate(self):
                if not dominates_one and hof_member.fitness.dominates(ind.fitness):
                    is_dominated = True
                    break
                elif ind.fitness.dominates(hof_member.fitness):
                    dominates_one = True
                    to_remove.append(i)
                elif ind.fitness == hof_member.fitness and self.similar(ind, hof_member):
                    has_twin = True
                    break

            for i in reversed(to_remove):
                self.remove(i)
            if not is_dominated and not has_twin:
                self.insert(ind)

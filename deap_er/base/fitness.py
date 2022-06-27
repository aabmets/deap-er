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
from typing import Sequence
from operator import mul, truediv


__all__ = ['Fitness']


# ====================================================================================== #
class Fitness:
    """
    A fitness object represents the measure of quality of a solution.
    The class attribute 'weights' tuple must be set before a Fitness object can be instantiated.
    """
    _err_msg_1 = "Can't instantiate 'Fitness', when class attribute 'weights' tuple is not set."
    _err_msg_2 = "The assigned values tuple must have the same length as " \
                 "the 'weights' attribute of the 'Fitness' class."

    weights: tuple = tuple()
    wvalues: tuple = tuple()

    # -------------------------------------------------------------------------------------- #
    def __init__(self, values: Sequence[int | float] = None) -> None:
        if not self.weights:
            raise TypeError(self._err_msg_1)
        if values:
            self.values = values

    # -------------------------------------------------------------------------------------- #
    @property
    def values(self) -> tuple:
        """Fitness values of the individual."""
        if self.is_valid():
            values = map(truediv, self.wvalues, self.weights)
            return tuple(values)
        return tuple()

    # -------------------------------------------------------------------------------------- #
    @values.setter
    def values(self, values: Sequence[int | float]) -> None:
        if len(values) != len(self.weights):
            raise TypeError(self._err_msg_2)
        wvalues = map(mul, values, self.weights)
        self.wvalues = tuple(wvalues)

    # -------------------------------------------------------------------------------------- #
    @values.deleter
    def values(self) -> None:
        self.wvalues = tuple()

    # -------------------------------------------------------------------------------------- #
    def dominates(self, other: Fitness, obj: slice = None) -> bool:
        """
        Returns true if each objective of *self* is not worse than the corresponding
        objective of the *other* and at least one objective is better.
        
        :param other: An instance of Fitness to test against.
        :param obj: A slice of objectives to test for domination.
                    If None, all objectives are tested.
        :return: True if other Fitness is worse.
        """
        obj = slice(None) if obj is None else obj
        zipper = list(zip(self.wvalues, other.wvalues))
        lesser = [a < b for a, b in zipper[obj]]
        equal = [a == b for a, b in zipper[obj]]
        if any(lesser) or all(equal):
            return False
        return True

    # -------------------------------------------------------------------------------------- #
    def is_valid(self) -> bool:
        """Returns True if the Fitness instance is valid."""
        a = len(self.weights)
        b = len(self.wvalues)
        return a == b and a > 0

    # -------------------------------------------------------------------------------------- #
    def __hash__(self):
        return hash(self.wvalues)

    def __gt__(self, other: Fitness) -> bool:
        return self.wvalues > other.wvalues

    def __ge__(self, other: Fitness) -> bool:
        return self.wvalues >= other.wvalues

    def __le__(self, other: Fitness) -> bool:
        return self.wvalues <= other.wvalues

    def __lt__(self, other: Fitness) -> bool:
        return self.wvalues < other.wvalues

    def __eq__(self, other: Fitness) -> bool:
        return self.wvalues == other.wvalues

    def __ne__(self, other: Fitness) -> bool:
        return self.wvalues != other.wvalues

    def __str__(self):
        return str(self.values)

    def __repr__(self):
        return '{0}.{1}({2})'.format(
            self.__module__,
            self.__class__.__name__,
            str(self.values)
        )

    def __deepcopy__(self, memo):
        copy = self.__class__()
        copy.wvalues = self.wvalues
        return copy

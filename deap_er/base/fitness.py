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
from deap_er.datatypes import SeqOfNum
from typing import Iterator
from operator import mul, truediv


__all__ = ['Fitness']


# ====================================================================================== #
class Fitness:
    """
    A fitness object measures the quality of a solution. The class attribute
    :attr:`weights` tuple must be set before a Fitness object can be instantiated.
    A fitness object can be instantiated without values, but the fitness object then
    remains invalid until the values have been set using the :attr:`values` property.

    Parameters:
        values (Optional[SeqOfNum]): The values of the fitness object, optional.
    """
    # -------------------------------------------------------- #
    weights: tuple = tuple()
    """
    The weights are used to compare the fitness of different individuals. They are 
    shared between all individuals of the same type. When subclassing :class:`Fitness`, 
    the weights class attribute must be a tuple of real numbers, where each element is 
    associated to an objective: a negative weight element corresponds to the minimization 
    and a positive weight to the maximization of the associated objective.
    """

    # -------------------------------------------------------- #
    wvalues: tuple = tuple()
    """
    Contains the weighted values of the fitness. These are obtained by
    multiplying the fitness values by the weights. It is generally unnecessary 
    to manipulate this attribute directly, as it's mostly used internally 
    by the Fitness comparison operators.
    """

    # -------------------------------------------------------- #
    def __init__(self, values: SeqOfNum = None) -> None:
        if not self.weights:
            raise TypeError(
                "Can't instantiate 'Fitness', when class "
                "attribute 'weights' tuple is not set."
            )
        if values:
            self.values = values

    # -------------------------------------------------------- #
    @property
    def values(self) -> tuple[float]:
        """
        Fitness values of the individual *(getter, setter and deleter)*.
        Setter accepts any :data:`SetItemSeq` object as input and the
        getter returns a tuple of floats. Deleter sets the internal
        :data:`wvalues` attribute to an empty tuple.

        Returns:
            tuple[float]: Fitness values of the individual.
        """
        if self.is_valid():
            values: Iterator = map(truediv, self.wvalues, self.weights)
            return tuple(values)
        return tuple()

    @values.setter
    def values(self, values: SeqOfNum) -> None:
        if len(values) != len(self.weights):
            raise TypeError(
                "The assigned values tuple must have the same length "
                "as the 'weights' attribute of the 'Fitness' class.")
        wvalues = map(mul, values, self.weights)
        self.wvalues = tuple(wvalues)

    @values.deleter
    def values(self) -> None:
        self.wvalues = tuple()

    # -------------------------------------------------------- #
    def dominates(self, other: Fitness, slc: slice = None) -> bool:
        """
        Returns true if each objective of *self* is not worse than
        the corresponding objective of the *other* and at least
        one objective of *self* is better.

        Parameters:
            other: An instance of Fitness to test against.
            slc: A slice of objectives to test for domination, optional.
        Returns:
            bool: True if *self* dominates *other*.
        """
        slc = slice(None) if slc is None else slc
        zipper = list(zip(self.wvalues, other.wvalues))
        lesser = [a < b for a, b in zipper[slc]]
        equal = [a == b for a, b in zipper[slc]]
        if any(lesser) or all(equal):
            return False
        return True

    # -------------------------------------------------------- #
    def is_valid(self) -> bool:
        """
        A Fitness instance is valid when the Fitness class attribute
        :attr:`weights` has been set and the instance property :attr:`values`
        has the same length as the :attr:`weights` attribute.

        Returns:
            bool: True if the Fitness instance is valid.
        """
        a = len(self.weights)
        b = len(self.wvalues)
        return a == b and a > 0

    # -------------------------------------------------------- #
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

    # -------------------------------------------------------- #
    def __str__(self):
        return str(self.values)

    def __repr__(self):
        return '{0}.{1}({2})'.format(
            self.__module__,
            self.__class__.__name__,
            str(self.values)
        )

    # -------------------------------------------------------- #
    def __deepcopy__(self, memo):
        copy = self.__class__()
        copy.wvalues = self.wvalues
        return copy

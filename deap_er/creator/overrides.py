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
from typing import Iterable
from copy import deepcopy
import array
import numpy


__all__ = ['_NumpyOverride', '_ArrayOverride']


# ====================================================================================== #
class _NumpyOverride(numpy.ndarray):
    """
    Class override for the *numpy.ndarray* class, because
    the *numpy.ndarray* class is problematic for DEAP-er.
    """
    # -------------------------------------------------------------------------------------- #
    @staticmethod
    def __new__(cls, iterable: Iterable) -> numpy.array:
        return numpy.array(list(iterable)).view(cls)

    # -------------------------------------------------------------------------------------- #
    def __deepcopy__(self, memo, *_, **__):
        copy = numpy.ndarray.copy(self)
        deep_copy = deepcopy(self.__dict__, memo)
        copy.__dict__.update(deep_copy)
        return copy

    # -------------------------------------------------------------------------------------- #
    def __setstate__(self, state, *_, **__):
        self.__dict__.update(state)

    # -------------------------------------------------------------------------------------- #
    def __reduce__(self):
        return self.__class__, (list(self),), self.__dict__


# ====================================================================================== #
class _ArrayOverride(array.array):
    """
    Class override for the *array.array* class, because
    the *array.array* class is problematic for DEAP-er.
    """
    # -------------------------------------------------------------------------------------- #
    @staticmethod
    def __new__(cls, typecode: str,  sequence: bytes | Iterable) -> array.array:
        return super().__new__(cls, typecode, sequence)

    # -------------------------------------------------------------------------------------- #
    def __deepcopy__(self, memo) -> object:
        cls = self.__class__
        copy = cls.__new__(cls, self.typecode, self)
        memo[id(self)] = copy
        deep_copy = deepcopy(self.__dict__, memo)
        copy.__dict__.update(deep_copy)
        return copy

    # -------------------------------------------------------------------------------------- #
    def __reduce__(self) -> tuple:
        return self.__class__, (list(self),), self.__dict__

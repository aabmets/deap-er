#
#   MIT License
#   
#   Copyright (c) 2022, Mattias Aabmets
#   
#   The contents of this file are subject to the terms and conditions defined in the License.
#   You may not use, modify, or distribute this file except in compliance with the License.
#   
#   SPDX-License-Identifier: MIT
#
from typing import Sequence
from copy import deepcopy
import array
import numpy


__all__ = ['_NumpyOverride', '_ArrayOverride']


# ====================================================================================== #
class _NumpyOverride(numpy.ndarray):
    """
    Class override for the 'numpy.ndarray' class, because
    the 'numpy.ndarray' class is problematic for DEAP-er.
    """
    @staticmethod
    def __new__(cls, seq: Sequence) -> numpy.array:
        return numpy.array(list(seq)).view(cls)

    def __deepcopy__(self, memo: dict, *_, **__):
        copy = numpy.ndarray.copy(self)
        dc = deepcopy(self.__dict__, memo)
        copy.__dict__.update(dc)
        return copy

    def __setstate__(self, state, *_, **__):
        self.__dict__.update(state)

    def __reduce__(self):
        return self.__class__, (list(self),), self.__dict__


# ====================================================================================== #
class _ArrayOverride(array.array):
    """
    Class override for the 'array.array' class, because
    the 'array.array' class is problematic for DEAP-er.
    """
    @staticmethod
    def __new__(cls, seq: Sequence) -> array.array:
        return super().__new__(cls, cls.typecode, seq)

    def __deepcopy__(self, memo: dict) -> object:
        cls = self.__class__
        copy = cls.__new__(cls, self)
        memo[id(self)] = copy
        dc = deepcopy(self.__dict__, memo)
        copy.__dict__.update(dc)
        return copy

    def __reduce__(self) -> tuple:
        return self.__class__, (list(self),), self.__dict__

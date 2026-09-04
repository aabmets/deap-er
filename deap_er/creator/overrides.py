#
#   Apache License 2.0
#
#   Copyright (c) 2022, Mattias Aabmets
#
#   The contents of this file are subject to the terms and conditions defined in the License.
#   You may not use, modify, or distribute this file except in compliance with the License.
#
#   SPDX-License-Identifier: Apache-2.0
#
from typing import Sequence
from copy import deepcopy
import array
import numpy


__all__ = ["_NumpyOverride", "_ArrayOverride"]


class _NumpyOverride(numpy.ndarray):
    """``numpy.ndarray`` subclass used by ``creator.create`` for array individuals."""

    @staticmethod
    def __new__(cls, seq: Sequence) -> numpy.array:
        """Build an instance from ``seq``."""
        return numpy.array(list(seq)).view(cls)

    def __deepcopy__(self, memo: dict, *_, **__):
        """Copy the array and its instance ``__dict__``."""
        copy = numpy.ndarray.copy(self)
        dc = deepcopy(self.__dict__, memo)
        copy.__dict__.update(dc)
        return copy

    def __setstate__(self, state, *_, **__):
        """Restore instance attributes from pickle ``state``."""
        self.__dict__.update(state)

    def __reduce__(self):
        """Return pickle reconstruction data."""
        return self.__class__, (list(self),), self.__dict__


class _ArrayOverride(array.array):
    """``array.array`` subclass used by ``creator.create`` for array individuals."""

    @staticmethod
    def __new__(cls, seq: Sequence) -> array.array:
        """Build an instance from ``seq`` using the subclass typecode."""
        return super().__new__(cls, cls.typecode, seq)

    def __deepcopy__(self, memo: dict) -> object:
        """Copy the array and its instance ``__dict__``."""
        cls = self.__class__
        copy = cls.__new__(cls, self)
        memo[id(self)] = copy
        dc = deepcopy(self.__dict__, memo)
        copy.__dict__.update(dc)
        return copy

    def __reduce__(self) -> tuple:
        """Return pickle reconstruction data."""
        return self.__class__, (list(self),), self.__dict__

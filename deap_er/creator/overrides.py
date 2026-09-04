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
import array
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, cast, override

import numpy

__all__ = ["_NumpyOverride", "_ArrayOverride"]


class _NumpyOverride(numpy.ndarray):
    """``numpy.ndarray`` subclass used by ``creator.create`` for array individuals."""

    @staticmethod
    def __new__(cls, seq: Sequence[Any]) -> numpy.ndarray:
        """Build an instance from ``seq``."""
        return numpy.array(list(seq)).view(cls)

    @override
    def __deepcopy__(self, memo: dict[int, Any], *_: Any, **__: Any) -> "_NumpyOverride":
        """Copy the array and its instance ``__dict__``."""
        copy = numpy.ndarray.copy(self)
        dc = deepcopy(self.__dict__, memo)
        copy.__dict__.update(dc)
        return copy

    @override
    def __setstate__(self, state: Any, *_: Any, **__: Any) -> None:
        """Restore instance attributes from pickle ``state``."""
        self.__dict__.update(state)

    @override
    def __reduce__(self) -> tuple[Any, ...]:
        """Return pickle reconstruction data."""
        return self.__class__, (list(self),), self.__dict__


class _ArrayOverride(array.array[Any]):
    """``array.array`` subclass used by ``creator.create`` for array individuals."""

    typecode: Any = "b"

    @staticmethod
    def __new__(cls, seq: Sequence[Any]) -> array.array[Any]:
        """Build an instance from ``seq`` using the subclass typecode."""
        return array.array.__new__(cls, str(cls.typecode), seq)

    @override
    def __deepcopy__(self, memo: dict[int, Any]) -> "_ArrayOverride":
        """Copy the array and its instance ``__dict__``."""
        cls = self.__class__
        copy = cast(_ArrayOverride, cls.__new__(cls, self))
        memo[id(self)] = copy
        dc = deepcopy(self.__dict__, memo)
        copy.__dict__.update(dc)
        return copy

    @override
    def __reduce__(self) -> tuple[Any, ...]:
        """Return pickle reconstruction data."""
        return self.__class__, (list(self),), self.__dict__

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
from collections.abc import Iterator
from typing import Any, Protocol

from .primitives import *

__all__ = ["GPIndividual", "GPMates", "GPMutant", "GPExprTypes", "GPTypedSets", "GPGraph"]


class GPIndividual(Protocol):
    """Prefix-tree individual used by genetic programming operators.

    :meta private:
    """

    fitness: Any
    root: Any
    height: int

    def search_subtree(self, begin: int) -> slice: ...
    def __getitem__(self, key: Any, /) -> Any: ...
    def __setitem__(self, key: int | slice, value: Any, /) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Any]: ...


type GPMates = tuple[GPIndividual, GPIndividual]
""":meta private:"""

type GPMutant = tuple[GPIndividual]
""":meta private:"""

type GPExprTypes = str | PrimitiveTree
""":meta private:"""

type GPTypedSets = list[PrimitiveSetTyped]
""":meta private:"""

type GPGraph = tuple[list[Any], list[Any], dict[Any, Any]]
""":meta private:"""

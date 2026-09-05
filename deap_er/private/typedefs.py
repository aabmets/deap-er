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
from collections.abc import Iterator, Sequence
from typing import Any, Protocol, Self

from .programming.primitives.primitive_set_typed import PrimitiveSetTyped
from .programming.primitives.primitive_tree import PrimitiveTree
from .records.hall_of_fame import HallOfFame, ParetoFront
from .records.logbook import Logbook
from .records.statistics import MultiStatistics, Statistics

__all__: list[str] = [
    "Individual",
    "Mates",
    "Mutant",
    "NumOrSeq",
    "GPIndividual",
    "GPMates",
    "GPMutant",
    "GPExprTypes",
    "GPTypedSets",
    "GPGraph",
    "EvoRecords",
    "EvoStats",
    "EvoAlgoResult",
]


class Individual(Protocol):
    """Sequence-like individual created by ``creator.create``.

    :meta private:
    """

    fitness: Any
    strategy: Any
    ps_: Any
    history_index: int

    def __getitem__(self, key: int | slice, /) -> Any: ...
    def __setitem__(self, key: int | slice, value: Any, /) -> None: ...
    def __delitem__(self, key: int | slice, /) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Any]: ...
    def copy(self) -> Self: ...


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


type Mates = tuple[Individual, Individual]
""":meta private:"""

type Mutant = tuple[Individual]
""":meta private:"""

type NumOrSeq = int | float | Sequence[int] | Sequence[float]
""":meta private:"""

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

type EvoRecords = HallOfFame | ParetoFront
""":meta private:"""

type EvoStats = Statistics | MultiStatistics
""":meta private:"""

type EvoAlgoResult = tuple[list[Any], Logbook]
""":meta private:"""

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

__all__ = ["Individual", "Mates", "Mutant", "NumOrSeq"]


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


type Mates = tuple[Individual, Individual]
""":meta private:"""

type Mutant = tuple[Individual]
""":meta private:"""

type NumOrSeq = int | float | Sequence[int] | Sequence[float]
""":meta private:"""

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
from collections.abc import Sequence
from typing import Any
import numpy
import array


__all__ = ["Individual", "Mates", "Mutant", "NumOrSeq"]


type Individual = list[Any] | array.array[Any] | numpy.ndarray
""":meta private:"""

type Mates = tuple[Individual, Individual]
""":meta private:"""

type Mutant = tuple[Individual]
""":meta private:"""

type NumOrSeq = int | float | Sequence[int] | Sequence[float]
""":meta private:"""

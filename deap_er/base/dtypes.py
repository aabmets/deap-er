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
from collections.abc import Sequence
from typing import Union, Tuple
import numpy
import array


__all__ = ['Individual', 'Mates', 'Mutant', 'NumOrSeq']


Individual = Union[list, array.array, numpy.ndarray]
""":meta private:"""

Mates = Tuple[Individual, Individual]
""":meta private:"""

Mutant = Tuple[Individual]
""":meta private:"""

NumOrSeq = Union[int, float, Sequence[int], Sequence[float]]
""":meta private:"""

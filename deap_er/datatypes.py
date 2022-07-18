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
from .records import *
from collections.abc import Sequence
from typing import Union, Tuple
import numpy
import array

Subscript = Union[dict, list, array.array, numpy.ndarray]
"""
| Any **mutable datatype** which can be used with the subscript operator **[ ]**.
| This includes the :data:`dict` , :data:`list` , :data:`array.array` , :data:`numpy.ndarray` and other
| datatypes as well as subclasses of these created by the :func:`~deap_er.creator.create` function.
"""

MatingResult = Tuple[Subscript, Subscript]
"""A pair of :data:`Subscript` datatypes."""

SeqOfNum = Union[Sequence[int], Sequence[float]]
"""A sequence of :data:`int` or :data:`float` values."""

NumOrSeq = Union[int, float, SeqOfNum]
"""Either an :data:`int` , :data:`float` or :data:`SeqOfNum`."""

Hof = Union[HallOfFame, ParetoFront]
"""Either a :data:`HallOfFame` or a :data:`ParetoFront` object."""

Stats = Union[Statistics, MultiStatistics]
"""Either a :data:`Statistics` or a :data:`MultiStatistics` object."""

AlgoResult = Tuple[list, Logbook]
"""
Returns:
    A tuple of :data:`list` and :data:`Logbook` objects.
"""

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
from __future__ import division
from deap_er.base import Individual
from functools import wraps
from typing import Callable


__all__ = [
    'bin2float', 'royal_road_1', 'royal_road_2',
    'chuang_f1', 'chuang_f2', 'chuang_f3'
]


# ====================================================================================== #
def bin2float(min_: float, max_: float, n_bits: int) -> Callable:
    """
    Returns a decorator, which converts a binary array into
    an array of floats where each float is composed of **n_bits**
    and has a value between **min** and **max** and returns the
    result of the decorated function.

    :param min_: Minimum value of the value range.
    :param max_: Maximum value of the value range.
    :param n_bits: Number of bits used to represent the float.
    :return: Decorated function.
    """
    def wrap(function):
        @wraps(function)
        def wrapped_function(individual, *args, **kwargs):
            nelem = len(individual) // n_bits
            decoded = [0] * nelem
            for i in range(nelem):
                start = i * n_bits
                stop = i * n_bits + n_bits
                values = individual[start:stop]
                mapper = map(str, values)
                gene = int("".join(mapper), 2)
                div = 2 ** n_bits - 1
                decoded[i] = min_ + ((gene / div) * (max_ - min_))
            return function(decoded, *args, **kwargs)
        return wrapped_function
    return wrap


# -------------------------------------------------------------------------------------- #
def royal_road_1(individual: Individual, order: int) -> int:
    """
    | Royal Road Function R1 as presented by Melanie Mitchell
    | in "An introduction to Genetic Algorithms".

    :param individual: The individual to be evaluated.
    :param order: The order of the royal road function.
    :return: The value of the royal road function.
    :type individual: :ref:`Individual <datatypes>`
    """
    nelem = len(individual) // order
    max_value = int(2 ** order - 1)
    total = 0
    for i in range(nelem):
        start = i * order
        stop = i * order + order
        values = individual[start:stop]
        mapper = map(str, values)
        gene = int("".join(mapper), 2)
        total += order * int(gene / max_value)
    return total


# -------------------------------------------------------------------------------------- #
def royal_road_2(individual: Individual, order: int) -> int:
    """
    | Royal Road Function R2 as presented by Melanie Mitchell
    | in "An introduction to Genetic Algorithms".

    :param individual: The individual to be evaluated.
    :param order: The order of the royal road function.
    :return: The value of the royal road function.
    :type individual: :ref:`Individual <datatypes>`
    """
    total = 0
    n_order = order
    while n_order < order ** 2:
        total += royal_road_1(individual, n_order)
        n_order *= 2
    return total


# -------------------------------------------------------------------------------------- #
def chuang_f1(individual: Individual) -> int:
    """
    | Binary deceptive function by Chung-Yao Chuang and Wen-Lian Hsu from
    | "Multivariate Multi-Model Approach for Globally Multimodal Problems".
    |
    | The function has two global optima in [1,1,...,1] and [0,0,...,0].
    | The individual has to be of 40 + 1 dimensions.

    :param individual: The individual to be evaluated.
    :return: The value of the deceptive function.
    :type individual: :ref:`Individual <datatypes>`
    """
    total = 0
    if individual[-1] == 0:
        for i in range(0, len(individual)-1, 4):
            total += _inv_trap(individual[i:i + 4])
    else:
        for i in range(0, len(individual)-1, 4):
            total += _trap(individual[i:i + 4])
    return total


# -------------------------------------------------------------------------------------- #
def chuang_f2(individual: Individual) -> int:
    """
    | Binary deceptive function by Chung-Yao Chuang and Wen-Lian Hsu from
    | "Multivariate Multi-Model Approach for Globally Multimodal Problems".
    |
    | The function has four global optima in [1,1,...,0,0], [0,0,...,1,1],
    | [1,1,...,1] and [0,0,...,0]. The individual has to be of 40 + 1 dimensions.

    :param individual: The individual to be evaluated.
    :return: The value of the deceptive function.
    :type individual: :ref:`Individual <datatypes>`
    """
    total = 0
    if individual[-2] == 0 and individual[-1] == 0:
        for i in range(0, len(individual)-2, 8):
            total += _inv_trap(individual[i:i + 4]) + _inv_trap(individual[i + 4:i + 8])
    elif individual[-2] == 0 and individual[-1] == 1:
        for i in range(0, len(individual)-2, 8):
            total += _inv_trap(individual[i:i + 4]) + _trap(individual[i + 4:i + 8])
    elif individual[-2] == 1 and individual[-1] == 0:
        for i in range(0, len(individual)-2, 8):
            total += _trap(individual[i:i + 4]) + _inv_trap(individual[i + 4:i + 8])
    else:
        for i in range(0, len(individual)-2, 8):
            total += _trap(individual[i:i + 4]) + _trap(individual[i + 4:i + 8])
    return total


# -------------------------------------------------------------------------------------- #
def chuang_f3(individual: Individual) -> int:
    """
    | Binary deceptive function by Chung-Yao Chuang and Wen-Lian Hsu from
    | "Multivariate Multi-Model Approach for Globally Multimodal Problems".
    |
    | The function has two global optima in [1,1,...,1] and [0,0,...,0].
    | The individual has to be of 40 + 1 dimensions.

    :param individual: The individual to be evaluated.
    :return: The value of the deceptive function.
    :type individual: :ref:`Individual <datatypes>`
    """
    total = 0
    if individual[-1] == 0:
        for i in range(0, len(individual)-1, 4):
            total += _inv_trap(individual[i:i + 4])
    else:
        for i in range(2, len(individual)-3, 4):
            total += _inv_trap(individual[i:i + 4])
        total += _trap(individual[-2:] + individual[:2])
    return total


# -------------------------------------------------------------------------------------- #
def _trap(individual: Individual) -> int:
    u = sum(individual)
    k = len(individual)
    return k if u == k else k - 1 - u


# -------------------------------------------------------------------------------------- #
def _inv_trap(individual: Individual) -> int:
    u = sum(individual)
    k = len(individual)
    return k if u == 0 else u - 1

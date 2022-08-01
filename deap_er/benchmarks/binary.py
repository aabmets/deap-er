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
from deap_er.base.dtypes import *


__all__ = [
    'bm_royal_road_1', 'bm_royal_road_2',
    'bm_chuang_f1', 'bm_chuang_f2', 'bm_chuang_f3'
]


# ====================================================================================== #
def bm_royal_road_1(individual: Individual, order: int) -> tuple[int]:
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
    return total,


# -------------------------------------------------------------------------------------- #
def bm_royal_road_2(individual: Individual, order: int) -> tuple[int]:
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
        total += bm_royal_road_1(individual, n_order)
        n_order *= 2
    return total,


# -------------------------------------------------------------------------------------- #
def bm_chuang_f1(individual: Individual) -> tuple[int]:
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
    return total,


# -------------------------------------------------------------------------------------- #
def bm_chuang_f2(individual: Individual) -> tuple[int]:
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
    return total,


# -------------------------------------------------------------------------------------- #
def bm_chuang_f3(individual: Individual) -> tuple[int]:
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
    return total,


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

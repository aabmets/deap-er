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
from deap_er._deprecated import deprecated
from collections.abc import Sequence
from itertools import repeat
import random
import math


__all__ = [
    'mut_gaussian', 'mutGaussian',
    'mut_polynomial_bounded', 'mutPolynomialBounded',
    'mut_shuffle_indexes', 'mutShuffleIndexes',
    'mut_flip_bit', 'mutFlipBit',
    'mut_uniform_int', 'mutUniformInt',
    'mut_es_log_normal', 'mutESLogNormal'
]


# ====================================================================================== #
def _check_bounds(individual, var1, var1_str, var2, var2_str) -> tuple:
    err_msg = '{0} must be at least the size of individual: {1} < {2}'
    size = len(individual)

    if not isinstance(var1, Sequence):
        var1 = repeat(var1, size)
    elif len(var1) < size:
        msg = err_msg.format(var1_str, len(var1), size)
        raise IndexError(msg)

    if not isinstance(var2, Sequence):
        var2 = repeat(var2, size)
    elif len(var2) < size:
        msg = err_msg.format(var2_str, len(var2), size)
        raise IndexError(msg)

    return var1, var2


# -------------------------------------------------------------------------------------- #
def mut_gaussian(individual, mu, sigma, ind_pb) -> tuple:
    mu, sigma = _check_bounds(individual, mu, 'mu', sigma, 'sigma')

    for i, m, s in zip(list(range(len(individual))), mu, sigma):
        if random.random() < ind_pb:
            individual[i] += random.gauss(m, s)

    return individual,


# -------------------------------------------------------------------------------------- #
def mut_polynomial_bounded(individual, eta, low, up, indpb):
    low, up = _check_bounds(individual, low, 'low', up, 'up')

    for i, xl, xu in zip(list(range(len(individual))), low, up):
        if random.random() <= indpb:
            x = individual[i]
            delta_1 = (x - xl) / (xu - xl)
            delta_2 = (xu - x) / (xu - xl)
            rand = random.random()
            mut_pow = 1.0 / (eta + 1.)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                delta_q = 1.0 - val ** mut_pow

            x = x + delta_q * (xu - xl)
            x = min(max(x, xl), xu)
            individual[i] = x

    return individual,


# -------------------------------------------------------------------------------------- #
def mut_shuffle_indexes(individual, ind_pb):
    size = len(individual)
    for i in range(size):
        if random.random() < ind_pb:
            swap_indx = random.randint(0, size - 2)
            if swap_indx >= i:
                swap_indx += 1
            individual[i], individual[swap_indx] = \
                individual[swap_indx], individual[i]

    return individual,


# -------------------------------------------------------------------------------------- #
def mut_flip_bit(individual, ind_pb) -> tuple:
    for i in range(len(individual)):
        if random.random() < ind_pb:
            individual[i] = type(individual[i])(not individual[i])

    return individual,


# -------------------------------------------------------------------------------------- #
def mut_uniform_int(individual, low, up, ind_pb) -> tuple:
    low, up = _check_bounds(individual, low, 'low', up, 'up')

    for i, xl, xu in zip(list(range(len(individual))), low, up):
        if random.random() < ind_pb:
            individual[i] = random.randint(xl, xu)

    return individual,


# -------------------------------------------------------------------------------------- #
def mut_es_log_normal(individual, c, ind_pb):
    size = len(individual)
    t = c / math.sqrt(2. * math.sqrt(size))
    t0 = c / math.sqrt(2. * size)
    n = random.gauss(0, 1)
    t0_n = t0 * n

    for indx in range(size):
        if random.random() < ind_pb:
            individual.strategy[indx] *= math.exp(t0_n + t * random.gauss(0, 1))
            individual[indx] += individual.strategy[indx] * random.gauss(0, 1)

    return individual,


# -------------------------------------------------------------------------------------- #
mutGaussian = deprecated('mutGaussian', mut_gaussian)
mutPolynomialBounded = deprecated('mutPolynomialBounded', mut_polynomial_bounded)
mutShuffleIndexes = deprecated('mutShuffleIndexes', mut_shuffle_indexes)
mutFlipBit = deprecated('mutFlipBit', mut_flip_bit)
mutUniformInt = deprecated('mutUniformInt', mut_uniform_int)
mutESLogNormal = deprecated('mutESLogNormal', mut_es_log_normal)

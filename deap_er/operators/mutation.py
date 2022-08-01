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
from deap_er.base.dtypes import *
from collections.abc import Sequence
from itertools import repeat
import random
import math


__all__ = [
    'mut_gaussian', 'mut_polynomial_bounded', 'mut_shuffle_indexes',
    'mut_flip_bit', 'mut_uniform_int', 'mut_es_log_normal'
]


# ====================================================================================== #
def _pre_process(name: str, var: NumOrSeq, size: int) -> Sequence:
    if not isinstance(var, Sequence):
        var = repeat(var, size)
    elif isinstance(var, Sequence) and len(var) < size:
        raise ValueError(
            f'Argument \'{name}\' must be at least the '
            f'size of the individual: {len(var)} < {size}'
        )
    return var


# -------------------------------------------------------------------------------------- #
def mut_gaussian(individual: Individual, mu: NumOrSeq,
                 sigma: NumOrSeq, mut_prob: float) -> Mutant:
    """
    Applies a gaussian mutation of mean **mu** and standard
    deviation **sigma** on the input individual.

    :param individual: The individual to be mutated.
    :param mu: The mean value of the gaussian mutation.
    :param sigma: The standard deviation of the gaussian mutation.
    :param mut_prob: The probability of mutating each attribute.
    :returns: A mutated individual.

    :type individual: :ref:`Individual <datatypes>`
    :type mu: :ref:`NumOrSeq <datatypes>`
    :type sigma: :ref:`NumOrSeq <datatypes>`
    :rtype: :ref:`Mutant <datatypes>`
    """
    size = len(individual)
    mu = _pre_process('mu', mu, size)
    sigma = _pre_process('sigma', sigma, size)

    idx = list(range(size))
    for i, m, s in zip(idx, mu, sigma):
        if random.random() < mut_prob:
            individual[i] += random.gauss(m, s)

    return individual,


# -------------------------------------------------------------------------------------- #
def mut_polynomial_bounded(individual: Individual,
                           eta: float, low: NumOrSeq,
                           up: NumOrSeq, mut_prob: float) -> Mutant:
    """
    Applies a polynomial mutation with a crowding
    degree of **eta** on the input individual.

    :param individual: The individual to be mutated.
    :param eta: The crowding degree of the crossover.
        Higher values produce children more similar to
        their parents, while smaller values produce
        children more divergent from their parents.
    :param low: The lower bound of the search space.
    :param up: The upper bound of the search space.
    :param mut_prob: The probability of mutating each attribute.
    :returns: A mutated individual.

    :type individual: :ref:`Individual <datatypes>`
    :type low: :ref:`NumOrSeq <datatypes>`
    :type up: :ref:`NumOrSeq <datatypes>`
    :rtype: :ref:`Mutant <datatypes>`
    """
    size = len(individual)
    low = _pre_process('low', low, size)
    up = _pre_process('up', up, size)

    idx = list(range(size))
    for i, xl, xu in zip(idx, low, up):
        if random.random() <= mut_prob:
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
def mut_shuffle_indexes(individual: Individual, mut_prob: float) -> Mutant:
    """
    Shuffles the attributes of the input individual.

    :param individual: The individual to be mutated.
    :param mut_prob: The probability of mutating each attribute.
    :returns: A mutated individual.

    :type individual: :ref:`Individual <datatypes>`
    :rtype: :ref:`Mutant <datatypes>`
    """
    size = len(individual)
    for i in range(size):
        if random.random() < mut_prob:
            swap_indx = random.randint(0, size - 2)
            if swap_indx >= i:
                swap_indx += 1
            individual[i], individual[swap_indx] = \
                individual[swap_indx], individual[i]

    return individual,


# -------------------------------------------------------------------------------------- #
def mut_flip_bit(individual: Individual, mut_prob: float) -> Mutant:
    """
    Flips the values of random attributes of the input individual.

    :param individual: The individual to be mutated.
    :param mut_prob: The probability of mutating each attribute.
    :returns: A mutated individual.

    :type individual: :ref:`Individual <datatypes>`
    :rtype: :ref:`Mutant <datatypes>`
    """
    for i in range(len(individual)):
        if random.random() < mut_prob:
            individual[i] = type(individual[i])(not individual[i])

    return individual,


# -------------------------------------------------------------------------------------- #
def mut_uniform_int(individual: Individual,
                    low: int, up: int,
                    mut_prob: float) -> Mutant:
    """
    | Mutates an individual by replacing attribute values with integers
    | chosen uniformly between the **low** and **up**, inclusively.

    :param individual: The individual to be mutated.
    :param low: The lower bound of the search space.
    :param up: The upper bound of the search space.
    :param mut_prob: The probability of mutating each attribute.
    :returns: A mutated individual.

    :type individual: :ref:`Individual <datatypes>`
    :rtype: :ref:`Mutant <datatypes>`
    """
    size = len(individual)
    low = _pre_process('low', low, size)
    up = _pre_process('up', up, size)

    idx = list(range(size))
    for i, xl, xu in zip(idx, low, up):
        if random.random() < mut_prob:
            individual[i] = random.randint(xl, xu)

    return individual,


# -------------------------------------------------------------------------------------- #
def mut_es_log_normal(individual: Individual,
                      learn_rate: float, mut_prob: float) -> Mutant:
    """
    Mutates an evolution strategy according to its *strategy* attribute.

    :param individual: The individual to be mutated.
    :param learn_rate: The learning rate of the evolution strategy. For
        an evolution strategy of (10, 100) the recommended value is 1.
    :param mut_prob: The probability of mutating each attribute.
    :returns: A mutated individual.

    :type individual: :ref:`Individual <datatypes>`
    :rtype: :ref:`Mutant <datatypes>`
    """
    size = len(individual)
    t = learn_rate / math.sqrt(2. * math.sqrt(size))
    t0 = learn_rate / math.sqrt(2. * size)
    n = random.gauss(0, 1)
    t0_n = t0 * n

    for indx in range(size):
        if random.random() < mut_prob:
            if hasattr(individual, 'strategy'):
                individual.strategy[indx] *= math.exp(t0_n + t * random.gauss(0, 1))
                individual[indx] += individual.strategy[indx] * random.gauss(0, 1)

    return individual,

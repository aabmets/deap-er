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


__all__ = [
    'cx_one_point', 'cxOnePoint',
    'cx_two_point', 'cxTwoPoint',
    'cx_uniform', 'cxUniform',
    'cx_partially_matched', 'cxPartialyMatched',
    'cx_uniform_partially_matched', 'cxUniformPartialyMatched',
    'cx_ordered', 'cxOrdered',
    'cx_blend', 'cxBlend',
    'cx_simulated_binary', 'cxSimulatedBinary',
    'cx_simulated_binary_bounded', 'cxSimulatedBinaryBounded',
    'cx_messy_one_point', 'cxMessyOnePoint',
    'cx_es_blend', 'cxESBlend',
    'cx_es_two_point', 'cxESTwoPoint'
]


# ====================================================================================== #
def cx_one_point(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cx_point = random.randint(1, size - 1)
    ind1[cx_point:], ind2[cx_point:] = ind2[cx_point:], ind1[cx_point:]

    return ind1, ind2


# -------------------------------------------------------------------------------------- #
def cx_two_point(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cx_point_1 = random.randint(1, size)
    cx_point_2 = random.randint(1, size - 1)

    if cx_point_2 >= cx_point_1:
        cx_point_2 += 1
    else:
        cx_point_1, cx_point_2 = cx_point_2, cx_point_1

    ind1[cx_point_1:cx_point_2] = ind2[cx_point_1:cx_point_2]
    ind2[cx_point_1:cx_point_2] = ind1[cx_point_1:cx_point_2]

    return ind1, ind2


# -------------------------------------------------------------------------------------- #
def cx_uniform(ind1, ind2, ind_pb):
    size = min(len(ind1), len(ind2))

    for i in range(size):
        if random.random() < ind_pb:
            ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2


# -------------------------------------------------------------------------------------- #
def cx_partially_matched(ind1, ind2):
    size = min(len(ind1), len(ind2))
    p1, p2 = [0] * size, [0] * size

    for i in range(size):
        p1[ind1[i]] = i
        p2[ind2[i]] = i

    cx_point_1 = random.randint(0, size)
    cx_point_2 = random.randint(0, size - 1)
    if cx_point_2 >= cx_point_1:
        cx_point_2 += 1
    else:
        cx_point_1, cx_point_2 = cx_point_2, cx_point_1

    for i in range(cx_point_1, cx_point_2):
        temp1 = ind1[i]
        temp2 = ind2[i]
        ind1[i], ind1[p1[temp2]] = temp2, temp1
        ind2[i], ind2[p2[temp1]] = temp1, temp2
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return ind1, ind2


# -------------------------------------------------------------------------------------- #
def cx_uniform_partially_matched(ind1, ind2, ind_pb):
    size = min(len(ind1), len(ind2))
    p1, p2 = [0] * size, [0] * size

    for i in range(size):
        p1[ind1[i]] = i
        p2[ind2[i]] = i

    for i in range(size):
        if random.random() < ind_pb:
            temp1 = ind1[i]
            temp2 = ind2[i]
            ind1[i], ind1[p1[temp2]] = temp2, temp1
            ind2[i], ind2[p2[temp1]] = temp1, temp2
            p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
            p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return ind1, ind2


# -------------------------------------------------------------------------------------- #
def cx_ordered(ind1, ind2):
    size = min(len(ind1), len(ind2))
    a, b = random.sample(list(range(size)), 2)
    if a > b:
        a, b = b, a

    holes1, holes2 = [True] * size, [True] * size
    for i in range(size):
        if i < a or i > b:
            holes1[ind2[i]] = False
            holes2[ind1[i]] = False

    temp1, temp2 = ind1, ind2
    k1, k2 = b + 1, b + 1

    for i in range(size):
        if not holes1[temp1[(i + b + 1) % size]]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not holes2[temp2[(i + b + 1) % size]]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1

    for i in range(a, b + 1):
        ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2


# -------------------------------------------------------------------------------------- #
def cx_blend(ind1, ind2, alpha):
    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        gamma = (1. + 2. * alpha) * random.random() - alpha
        ind1[i] = (1. - gamma) * x1 + gamma * x2
        ind2[i] = gamma * x1 + (1. - gamma) * x2

    return ind1, ind2


# -------------------------------------------------------------------------------------- #
def cx_simulated_binary(ind1, ind2, eta):
    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        rand = random.random()

        if rand <= 0.5:
            beta = 2. * rand
        else:
            beta = 1. / (2. * (1. - rand))

        beta **= 1. / (eta + 1.)
        ind1[i] = 0.5 * (((1 + beta) * x1) + ((1 - beta) * x2))
        ind2[i] = 0.5 * (((1 - beta) * x1) + ((1 + beta) * x2))

    return ind1, ind2


# -------------------------------------------------------------------------------------- #
def cx_simulated_binary_bounded(ind1, ind2, eta, low, up):
    err_msg = '{1} must be at least the size of the shorter individual: {2} < {3}'
    size = min(len(ind1), len(ind2))

    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError(err_msg.format('low', len(low), size))

    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError(err_msg.format('up', len(up), size))

    for i, xl, xu in zip(list(range(size)), low, up):
        if random.random() <= 0.5:
            if abs(ind1[i] - ind2[i]) > 1e-14:
                x1 = min(ind1[i], ind2[i])
                x2 = max(ind1[i], ind2[i])
                rand = random.random()

                beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)

                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)

                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                c1 = min(max(c1, xl), xu)
                c2 = min(max(c2, xl), xu)

                if random.random() <= 0.5:
                    ind1[i] = c2
                    ind2[i] = c1
                else:
                    ind1[i] = c1
                    ind2[i] = c2

    return ind1, ind2


# -------------------------------------------------------------------------------------- #
def cx_messy_one_point(ind1, ind2):
    cx_point_1 = random.randint(0, len(ind1))
    cx_point_2 = random.randint(0, len(ind2))
    ind1[cx_point_1:], ind2[cx_point_2:] = ind2[cx_point_2:], ind1[cx_point_1:]

    return ind1, ind2


# -------------------------------------------------------------------------------------- #
def cx_es_blend(ind1, ind2, alpha):
    zipper = zip(ind1, ind1.strategy, ind2, ind2.strategy)
    for i, (x1, s1, x2, s2) in enumerate(zipper):

        gamma = (1. + 2. * alpha) * random.random() - alpha
        ind1[i] = (1. - gamma) * x1 + gamma * x2
        ind2[i] = gamma * x1 + (1. - gamma) * x2

        gamma = (1. + 2. * alpha) * random.random() - alpha
        ind1.strategy[i] = (1. - gamma) * s1 + gamma * s2
        ind2.strategy[i] = gamma * s1 + (1. - gamma) * s2

    return ind1, ind2


# -------------------------------------------------------------------------------------- #
def cx_es_two_point(ind1, ind2):
    size = min(len(ind1), len(ind2))

    pt1 = random.randint(1, size)
    pt2 = random.randint(1, size - 1)

    if pt2 >= pt1:
        pt2 += 1
    else:
        pt1, pt2 = pt2, pt1

    ind1[pt1:pt2], ind2[pt1:pt2] = ind2[pt1:pt2], ind1[pt1:pt2]
    ind1.strategy[pt1:pt2], ind2.strategy[pt1:pt2] = \
        ind2.strategy[pt1:pt2], ind1.strategy[pt1:pt2]

    return ind1, ind2


# -------------------------------------------------------------------------------------- #
cxOnePoint = deprecated('cxOnePoint', cx_one_point)
cxTwoPoint = deprecated('cxTwoPoint', cx_two_point)
cxUniform = deprecated('cxUniform', cx_uniform)
cxPartialyMatched = deprecated('cxPartialyMatched', cx_partially_matched)
cxUniformPartialyMatched = deprecated('cxUniformPartialyMatched', cx_uniform_partially_matched)
cxOrdered = deprecated('cxOrdered', cx_ordered)
cxBlend = deprecated('cxBlend', cx_blend)
cxSimulatedBinary = deprecated('cxSimulatedBinary', cx_simulated_binary)
cxSimulatedBinaryBounded = deprecated('cxSimulatedBinaryBounded', cx_simulated_binary_bounded)
cxMessyOnePoint = deprecated('cxMessyOnePoint', cx_messy_one_point)
cxESBlend = deprecated('cxESBlend', cx_es_blend)
cxESTwoPoint = deprecated('cxESTwoPoint', cx_es_two_point)

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
from operator import attrgetter
import random


__all__ = [
    'sel_random', 'selRandom',
    'sel_best', 'selBest',
    'sel_worst', 'selWorst',
    'sel_roulette', 'selRoulette',
    'sel_stochastic_universal_sampling', 'selStochasticUniversalSampling'
]


# ====================================================================================== #
def sel_random(individuals, k):
    return [random.choice(individuals) for _ in range(k)]


# -------------------------------------------------------------------------------------- #
def sel_best(individuals, k, fit_attr="fitness"):
    return sorted(individuals, key=attrgetter(fit_attr), reverse=True)[:k]


# -------------------------------------------------------------------------------------- #
def sel_worst(individuals, k, fit_attr="fitness"):
    return sorted(individuals, key=attrgetter(fit_attr))[:k]


# -------------------------------------------------------------------------------------- #
def sel_roulette(individuals, k, fit_attr="fitness"):
    sorted_individuals = sorted(individuals, key=attrgetter(fit_attr), reverse=True)
    sum_fits = sum(getattr(ind, fit_attr).values[0] for ind in individuals)
    chosen = []
    for i in range(k):
        u = random.random() * sum_fits
        sum_ = 0
        for ind in sorted_individuals:
            sum_ += getattr(ind, fit_attr).values[0]
            if sum_ > u:
                chosen.append(ind)
                break

    return chosen


# -------------------------------------------------------------------------------------- #
def sel_stochastic_universal_sampling(individuals, k, fit_attr="fitness"):
    s_inds = sorted(individuals, key=attrgetter(fit_attr), reverse=True)
    sum_fits = sum(getattr(ind, fit_attr).values[0] for ind in individuals)

    distance = sum_fits / float(k)
    start = random.uniform(0, distance)
    points = [start + i*distance for i in range(k)]

    chosen = []
    for p in points:
        i = 0
        sum_ = getattr(s_inds[i], fit_attr).values[0]
        while sum_ < p:
            i += 1
            sum_ += getattr(s_inds[i], fit_attr).values[0]
        chosen.append(s_inds[i])

    return chosen


# -------------------------------------------------------------------------------------- #
selRandom = deprecated('selRandom', sel_random)
selBest = deprecated('selBest', sel_best)
selWorst = deprecated('selWorst', sel_worst)
selRoulette = deprecated('selRoulette', sel_roulette)
selStochasticUniversalSampling = deprecated('selStochasticUniversalSampling', sel_stochastic_universal_sampling)

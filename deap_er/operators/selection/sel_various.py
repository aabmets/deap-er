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
from deap_er.datatypes import SetItemSeq
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
def sel_random(individuals: SetItemSeq, count: int) -> list:
    """
    Selects *count* individuals randomly.

    :param individuals: A list of individuals to select from.
    :param count: The number of individuals to select.
    :returns: A list of selected individuals.
    """
    return [random.choice(individuals) for _ in range(count)]


# -------------------------------------------------------------------------------------- #
def sel_best(individuals: SetItemSeq, count: int,
             fit_attr: str = "fitness") -> list:
    """
    Selects the best *count* individuals among the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param count: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as selection criterion.
    :returns: A list of selected individuals.
    """
    key = attrgetter(fit_attr)
    return sorted(individuals, key=key, reverse=True)[:count]


# -------------------------------------------------------------------------------------- #
def sel_worst(individuals: SetItemSeq, count: int,
              fit_attr: str = "fitness") -> list:
    """
    Selects the worst *count* individuals among the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param count: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as selection criterion.
    :returns: A list of selected individuals.
    """
    key = attrgetter(fit_attr)
    return sorted(individuals, key=key)[:count]


# -------------------------------------------------------------------------------------- #
def sel_roulette(individuals: SetItemSeq, count: int,
                 fit_attr: str = "fitness") -> list:
    """
    Select *k* individuals from the input *individuals* using *k*
    spins of a roulette. The selection is made by looking only at the
    first objective of each individual. The list returned contains
    references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param count: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as selection criterion.
    :returns: A list of selected individuals.
    """
    key = attrgetter(fit_attr)
    sorted_ = sorted(individuals, key=key, reverse=True)
    sum_fits = sum(getattr(ind, fit_attr).values[0] for ind in individuals)
    chosen = []
    for i in range(count):
        u = random.random() * sum_fits
        sum_ = 0
        for ind in sorted_:
            sum_ += getattr(ind, fit_attr).values[0]
            if sum_ > u:
                chosen.append(ind)
                break

    return chosen


# -------------------------------------------------------------------------------------- #
def sel_stochastic_universal_sampling(individuals: SetItemSeq, count: int,
                                      fit_attr: str = "fitness") -> list:
    """
    Selects the *k* individuals among the input *individuals*.
    The selection is made by using a single random value to sample
    all the individuals by choosing them at evenly spaced intervals.
    The list returned contains references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param count: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as selection criterion.
    :returns: A list of selected individuals.
    """
    key = attrgetter(fit_attr)
    sorted_ = sorted(individuals, key=key, reverse=True)
    sum_fits = sum(getattr(ind, fit_attr).values[0] for ind in individuals)

    distance = sum_fits / float(count)
    start = random.uniform(0, distance)
    points = [start + i * distance for i in range(count)]

    chosen = []
    for p in points:
        i = 0
        sum_ = getattr(sorted_[i], fit_attr).values[0]
        while sum_ < p:
            i += 1
            sum_ += getattr(sorted_[i], fit_attr).values[0]
        chosen.append(sorted_[i])

    return chosen


# -------------------------------------------------------------------------------------- #
selRandom = deprecated('selRandom', sel_random)
selBest = deprecated('selBest', sel_best)
selWorst = deprecated('selWorst', sel_worst)
selRoulette = deprecated('selRoulette', sel_roulette)
selStochasticUniversalSampling = deprecated('selStochasticUniversalSampling', sel_stochastic_universal_sampling)

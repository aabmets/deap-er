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
from deap_er.datatypes import Individual
from operator import attrgetter
import random


__all__ = [
    'sel_random', 'sel_best', 'sel_worst', 'sel_roulette',
    'sel_stochastic_universal_sampling'
]


# ====================================================================================== #
def sel_random(individuals: Individual, sel_count: int) -> list:
    """
    Selects *count* individuals randomly.

    Parameters:
        individuals: A list of individuals to select from.
        sel_count: The number of individuals to select.
    Returns:
        A list of selected individuals.
    """
    return [random.choice(individuals) for _ in range(sel_count)]


# -------------------------------------------------------------------------------------- #
def sel_best(individuals: Individual, sel_count: int,
             fit_attr: str = "fitness") -> list:
    """
    Selects the best *count* individuals among the input *individuals*.

    Parameters:
        individuals: A list of individuals to select from.
        sel_count: The number of individuals to select.
        fit_attr: The attribute of individuals to use as the selection criterion.
    Returns:
        A list of selected individuals.
    """
    key = attrgetter(fit_attr)
    return sorted(individuals, key=key, reverse=True)[:sel_count]


# -------------------------------------------------------------------------------------- #
def sel_worst(individuals: Individual, sel_count: int,
              fit_attr: str = "fitness") -> list:
    """
    Selects the worst *count* individuals among the input *individuals*.

    Parameters:
        individuals: A list of individuals to select from.
        sel_count: The number of individuals to select.
        fit_attr: The attribute of individuals to use as the selection criterion.
    Returns:
        A list of selected individuals.
    """
    key = attrgetter(fit_attr)
    return sorted(individuals, key=key)[:sel_count]


# -------------------------------------------------------------------------------------- #
def sel_roulette(individuals: Individual, sel_count: int,
                 fit_attr: str = "fitness") -> list:
    """
    Select *k* individuals from the input *individuals* using *k*
    spins of a roulette. The selection is made by looking only at the
    first objective of each individual. The list returned contains
    references to the input *individuals*.

    Parameters:
        individuals: A list of individuals to select from.
        sel_count: The number of individuals to select.
        fit_attr: The attribute of individuals to use as the selection criterion.
    Returns:
        A list of selected individuals.
    """
    key = attrgetter(fit_attr)
    sorted_ = sorted(individuals, key=key, reverse=True)
    sum_fits = sum(getattr(ind, fit_attr).values[0] for ind in individuals)
    chosen = []
    for i in range(sel_count):
        u = random.random() * sum_fits
        sum_ = 0
        for ind in sorted_:
            sum_ += getattr(ind, fit_attr).values[0]
            if sum_ > u:
                chosen.append(ind)
                break

    return chosen


# -------------------------------------------------------------------------------------- #
def sel_stochastic_universal_sampling(individuals: Individual, sel_count: int,
                                      fit_attr: str = "fitness") -> list:
    """
    Selects the *k* individuals among the input *individuals*.
    The selection is made by using a single random value to sample
    all the individuals by choosing them at evenly spaced intervals.
    The list returned contains references to the input *individuals*.

    Parameters:
        individuals: A list of individuals to select from.
        sel_count: The number of individuals to select.
        fit_attr: The attribute of individuals to use as the selection criterion.
    Returns:
        A list of selected individuals.
    """
    key = attrgetter(fit_attr)
    sorted_ = sorted(individuals, key=key, reverse=True)
    sum_fits = sum(getattr(ind, fit_attr).values[0] for ind in individuals)

    distance = sum_fits / float(sel_count)
    start = random.uniform(0, distance)
    points = [start + i * distance for i in range(sel_count)]

    chosen = []
    for p in points:
        i = 0
        sum_ = getattr(sorted_[i], fit_attr).values[0]
        while sum_ < p:
            i += 1
            sum_ += getattr(sorted_[i], fit_attr).values[0]
        chosen.append(sorted_[i])

    return chosen

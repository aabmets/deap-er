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
from deap_er._datatypes import SetItemSeq
import numpy as np
import random


__all__ = [
    'sel_lexicase', 'selLexicase',
    'sel_epsilon_lexicase', 'selEpsilonLexicase'
]


# ====================================================================================== #
def sel_lexicase(individuals: SetItemSeq, count) -> list:
    """
    Returns an individual that does the best on the fitness
    cases when considered one at a time in random order.

    :param individuals: A list of individuals to select from.
    :param count: The number of individuals to select.
    :returns: A list of selected individuals.
    """
    selected = []
    for i in range(count):
        fit_weights = individuals[0].fitness.weights
        candidates = individuals
        cases = list(range(len(individuals[0].fitness.values)))
        random.shuffle(cases)
        while len(cases) > 0 and len(candidates) > 1:
            f = min
            if fit_weights[cases[0]] > 0:
                f = max
            f_vals = [x.fitness.values[cases[0]] for x in candidates]
            best_val = f(f_vals)
            candidates = [x for x in candidates if x.fitness.values[cases[0]] == best_val]
            cases.pop(0)
        choice = random.choice(candidates)
        selected.append(choice)
    return selected


# -------------------------------------------------------------------------------------- #
def sel_epsilon_lexicase(individuals: SetItemSeq, count: int,
                         epsilon: float = None) -> list:
    """
    Returns an individual that does the best on the fitness cases
    when considered one at a time in random order.

    :param individuals: A list of individuals to select from.
    :param count: The number of individuals to select.
    :param epsilon: The epsilon parameter, optional.
        If not provided, the epsilon parameter is automatically
        calculated from the median of fitness values.
    :returns: A list of selected individuals.
    """
    selected = []
    for i in range(count):
        fit_weights = individuals[0].fitness.weights
        cases = list(range(len(individuals[0].fitness.values)))
        random.shuffle(cases)
        candidates = individuals
        while len(cases) > 0 and len(candidates) > 1:
            f_vals = [x.fitness.values[cases[0]] for x in candidates]
            if not epsilon:
                median = np.median(f_vals)
                epsilon = np.median([abs(x - median) for x in f_vals])
            if fit_weights[cases[0]] > 0:
                best_val = max(f_vals)
                min_val = best_val - epsilon
                candidates = [x for x in candidates if x.fitness.values[cases[0]] >= min_val]
            else:
                best_val = min(f_vals)
                max_val = best_val + epsilon
                candidates = [x for x in candidates if x.fitness.values[cases[0]] <= max_val]
            cases.pop(0)

        choice = random.choice(candidates)
        selected.append(choice)
    return selected


# -------------------------------------------------------------------------------------- #
selLexicase = deprecated('selLexicase', sel_lexicase)
selEpsilonLexicase = deprecated('selEpsilonLexicase', sel_epsilon_lexicase)

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
import numpy as np
import random


__all__ = [
    'sel_lexicase', 'selLexicase',
    'sel_epsilon_lexicase', 'selEpsilonLexicase',
    'sel_automatic_epsilon_lexicase', 'selAutomaticEpsilonLexicase'
]


# ====================================================================================== #
def sel_lexicase(individuals, k):
    selected_individuals = []

    for i in range(k):
        fit_weights = individuals[0].fitness.weights

        candidates = individuals
        cases = list(range(len(individuals[0].fitness.values)))
        random.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            f = min
            if fit_weights[cases[0]] > 0:
                f = max

            best_val_for_case = f([x.fitness.values[cases[0]] for x in candidates])

            candidates = list([x for x in candidates if x.fitness.values[cases[0]] == best_val_for_case])
            cases.pop(0)

        selected_individuals.append(random.choice(candidates))

    return selected_individuals


# -------------------------------------------------------------------------------------- #
def sel_epsilon_lexicase(individuals, k, epsilon):
    selected_individuals = []

    for i in range(k):
        fit_weights = individuals[0].fitness.weights

        candidates = individuals
        cases = list(range(len(individuals[0].fitness.values)))
        random.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            if fit_weights[cases[0]] > 0:
                best_val_for_case = max([x.fitness.values[cases[0]] for x in candidates])
                min_val_to_survive_case = best_val_for_case - epsilon
                candidates = list([x for x in candidates if x.fitness.values[cases[0]] >= min_val_to_survive_case])
            else:
                best_val_for_case = min([x.fitness.values[cases[0]] for x in candidates])
                max_val_to_survive_case = best_val_for_case + epsilon
                candidates = list([x for x in candidates if x.fitness.values[cases[0]] <= max_val_to_survive_case])

            cases.pop(0)

        selected_individuals.append(random.choice(candidates))

    return selected_individuals


# -------------------------------------------------------------------------------------- #
def sel_automatic_epsilon_lexicase(individuals, k):
    selected_individuals = []

    for i in range(k):
        fit_weights = individuals[0].fitness.weights

        candidates = individuals
        cases = list(range(len(individuals[0].fitness.values)))
        random.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            errors_for_this_case = [x.fitness.values[cases[0]] for x in candidates]
            median_val = np.median(errors_for_this_case)
            median_absolute_deviation = np.median([abs(x - median_val) for x in errors_for_this_case])
            if fit_weights[cases[0]] > 0:
                best_val_for_case = max(errors_for_this_case)
                min_val_to_survive = best_val_for_case - median_absolute_deviation
                candidates = list([x for x in candidates if x.fitness.values[cases[0]] >= min_val_to_survive])
            else:
                best_val_for_case = min(errors_for_this_case)
                max_val_to_survive = best_val_for_case + median_absolute_deviation
                candidates = list([x for x in candidates if x.fitness.values[cases[0]] <= max_val_to_survive])

            cases.pop(0)

        selected_individuals.append(random.choice(candidates))

    return selected_individuals


# -------------------------------------------------------------------------------------- #
selLexicase = deprecated('selLexicase', sel_lexicase)
selEpsilonLexicase = deprecated('selEpsilonLexicase', sel_epsilon_lexicase)
selAutomaticEpsilonLexicase = deprecated('selAutomaticEpsilonLexicase', sel_automatic_epsilon_lexicase)

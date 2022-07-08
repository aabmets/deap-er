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
from deap_er.utils import Logbook, Statistics, HallOfFame
from deap_er.base.toolbox import Toolbox
from .variation import *


__all__ = ['ea_mu_comma_lambda', 'eaMuCommaLambda']


# -------------------------------------------------------------------------------------- #
def ea_mu_comma_lambda(toolbox: Toolbox,
                       population: list,
                       mu: int,
                       lambda_: int,
                       cx_prob: float,
                       mut_prob: float,
                       n_gen: int,
                       hof: HallOfFame = None,
                       stats: Statistics = None,
                       verbose: int = __debug__) -> tuple[list, Logbook]:
    """
    An evolutionary algorithm. This function expects the *mate*, *mutate*,
    *select* and *evaluate* operators to be registered in the toolbox.

    :param toolbox: A Toolbox which contains the evolution operators.
    :param population: A list of individuals to vary.
    :param mu: The number of individuals to select for the next generation.
    :param lambda_: The number of children to produce at each generation.
    :param cx_prob: The probability of mating two individuals.
    :param mut_prob: The probability of mutating an individual.
    :param n_gen: The number of generations to compute.
    :param hof: A HallOfFame object, optional.
    :param stats: A Statistics object, optional.
    :param verbose: Whether to print debug messages, optional.
    :returns: Tuple of the final population and the logbook.
    """
    if lambda_ < mu:
        msg = 'child_count must be larger than or equal to ngen_ind_count.'
        raise ValueError(msg)

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitness = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitness):
        ind.fitness.values = fit

    if hof is not None:
        hof.update(population)

    logbook = Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, n_gen + 1):
        offspring = var_or(toolbox, population, lambda_, cx_prob, mut_prob)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        if hof is not None:
            hof.update(offspring)

        population[:] = toolbox.select(offspring, mu)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


# -------------------------------------------------------------------------------------- #
eaMuCommaLambda = deprecated('eaMuCommaLambda', ea_mu_comma_lambda)

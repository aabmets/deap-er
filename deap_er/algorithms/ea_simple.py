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
from deap_er.records import Logbook, Hof, Stats, AlgoResult
from deap_er.base import Toolbox
from .variation import *


__all__ = ['ea_simple']


# ====================================================================================== #
def ea_simple(toolbox: Toolbox, population: list,
              generations: int, cx_prob: float,
              mut_prob: float, hof: Hof = None,
              stats: Stats = None, verbose: bool = False) -> AlgoResult:
    """
    An evolutionary algorithm. This function expects the *'mate'*, *'mutate'*,
    *'select'* and *'evaluate'* operators to be registered in the toolbox.

    :param toolbox: A Toolbox which contains the evolution operators.
    :param population: A list of individuals to evolve.
    :param generations: The number of generations to compute.
    :param cx_prob: The probability of mating two individuals.
    :param mut_prob: The probability of mutating an individual.
    :param hof: A HallOfFame or a ParetoFront object, optional.
    :param stats: A Statistics or a MultiStatistics object, optional.
    :param verbose: Whether to print debug messages, optional.
    :return: The final population and the logbook.

    :type hof: :ref:`Hof <datatypes>`
    :type stats: :ref:`Stats <datatypes>`
    :rtype: :ref:`AlgoResult <datatypes>`
    """
    logbook = Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.is_valid()]
    fitness = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitness):
        ind.fitness.values = fit

    if hof is not None:
        hof.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)

    if verbose:
        print(logbook.stream)

    for gen in range(1, generations + 1):
        offspring = toolbox.select(population, len(population))
        offspring = var_and(toolbox, offspring, cx_prob, mut_prob)

        invalid_ind = [ind for ind in offspring if not ind.fitness.is_valid()]
        fitness = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        if hof is not None:
            hof.update(offspring)

        population[:] = offspring
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if verbose:
            print(logbook.stream)

    return population, logbook

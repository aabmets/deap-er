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
from deap_er.records.dtypes import *
from deap_er.records import Logbook
from deap_er.base import Toolbox
from .variation import *


__all__ = ['ea_mu_comma_lambda']


# ====================================================================================== #
def ea_mu_comma_lambda(toolbox: Toolbox, population: list,
                       generations: int, offsprings: int,
                       survivors: int, cx_prob: float,
                       mut_prob: float, hof: Hof = None,
                       stats: Stats = None, verbose: bool = False) -> AlgoResult:
    """
    An evolutionary algorithm. This function expects the *'mate'*, *'mutate'*,
    *'select'* and *'evaluate'* operators to be registered in the toolbox.
    The survivors are selected only from the offspring population.

    :param toolbox: A Toolbox which contains the evolution operators.
    :param population: A list of individuals to evolve.
    :param generations: The number of generations to compute.
    :param offsprings: The number of individuals to produce at each generation.
    :param survivors: The number of individuals to select from the offspring.
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
    if survivors > offsprings:  # pragma: no cover
        offsprings, survivors = survivors, offsprings

    logbook = Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    for gen in range(1, generations + 1):
        offspring = var_or(toolbox, population, offsprings, cx_prob, mut_prob)

        invalids = [ind for ind in offspring if not ind.fitness.is_valid()]
        fitness = toolbox.map(toolbox.evaluate, invalids)
        for ind, fit in zip(invalids, fitness):
            ind.fitness.values = fit

        population[:] = toolbox.select(offspring, survivors)

        if hof is not None:
            hof.update(offspring)
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalids), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

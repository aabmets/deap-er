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
from deap_er.datatypes import Hof, Stats, AlgoResult
from deap_er.base.toolbox import Toolbox
from deap_er.records import Logbook
from .variation import *


__all__ = ['ea_mu_comma_lambda']


# ====================================================================================== #
def ea_mu_comma_lambda(toolbox: Toolbox, population: list,
                       generations: int, survivors: int,
                       offsprings: int, cx_prob: float,
                       mut_prob: float, hof: Hof = None,
                       stats: Stats = None, verbose: bool = False) -> AlgoResult:
    """
    An evolutionary algorithm. This function expects the 'mate', 'mutate', 'select'
    and 'evaluate' operators to be registered in the toolbox. The survivors are
    selected only from the offspring population.

    Parameters:
        toolbox: A Toolbox which contains the evolution operators.
        population: A list of individuals to evolve.
        generations: The number of generations to compute.
        survivors: The number of individuals to select from the offspring.
        offsprings: The number of individuals to produce at each generation.
        cx_prob: The probability of mating two individuals.
        mut_prob: The probability of mutating an individual.
        hof: A HallOfFame or a ParetoFront object, optional.
        stats: A Statistics or a MultiStatistics object, optional.
        verbose: Whether to print debug messages, optional.
    Returns:
        The final population and the logbook.
    """
    if offsprings < survivors:
        raise ValueError(
            '\'offsprings\' must be greater than or equal to \'survivors\'.'
        )

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

    for gen in range(1, generations + 1):
        offspring = var_or(toolbox, population, offsprings, cx_prob, mut_prob)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        if hof is not None:
            hof.update(offspring)

        population[:] = toolbox.select(offspring, survivors)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

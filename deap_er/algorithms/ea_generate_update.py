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
from deap_er.records import Logbook, Statistics, HallOfFame
from deap_er.base.toolbox import Toolbox


__all__ = ['ea_generate_update', 'eaGenerateUpdate']


# -------------------------------------------------------------------------------------- #
def ea_generate_update(toolbox: Toolbox,
                       n_gen: int,
                       hof: HallOfFame = None,
                       stats: Statistics = None,
                       verbose: bool = False) -> tuple[list, Logbook]:
    """
    An evolutionary algorithm. This function expects the *generate*, *update*,
    and *evaluate* operators to be registered in the toolbox.

    :param toolbox: A Toolbox which contains the evolution operators.
    :param n_gen: The number of generations to compute.
    :param hof: A HallOfFame object, optional.
    :param stats: A Statistics object, optional.
    :param verbose: Whether to print debug messages, optional.
    :returns: Tuple of the final population and the logbook.
    """
    logbook = Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    population = None

    for gen in range(n_gen):
        population = toolbox.generate()
        fitness = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitness):
            ind.fitness.values = fit

        if hof is not None:
            hof.update(population)

        toolbox.update(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


# -------------------------------------------------------------------------------------- #
eaGenerateUpdate = deprecated('eaGenerateUpdate', ea_generate_update)

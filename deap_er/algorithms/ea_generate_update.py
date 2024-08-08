#
#   MIT License
#   
#   Copyright (c) 2022, Mattias Aabmets
#   
#   The contents of this file are subject to the terms and conditions defined in the License.
#   You may not use, modify, or distribute this file except in compliance with the License.
#   
#   SPDX-License-Identifier: MIT
#
from deap_er.records.dtypes import *
from deap_er.records import Logbook
from deap_er.base import Toolbox


__all__ = ['ea_generate_update']


# ====================================================================================== #
def ea_generate_update(toolbox: Toolbox, generations: int,
                       hof: Hof = None, stats: Stats = None,
                       verbose: bool = False) -> AlgoResult:
    """
    An evolutionary algorithm. This function expects the *'generate'*,
    *'update'*, and *'evaluate'* operators to be registered in the toolbox.

    :param toolbox: A Toolbox which contains the evolution operators.
    :param generations: The number of generations to compute.
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

    population = None
    for gen in range(generations):
        population = toolbox.generate()

        fitness = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitness):
            ind.fitness.values = fit

        toolbox.update(population)

        if hof is not None:
            hof.update(population)
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

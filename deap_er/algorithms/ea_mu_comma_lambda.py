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

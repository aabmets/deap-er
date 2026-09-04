#
#   Apache License 2.0
#
#   Copyright (c) 2022, Mattias Aabmets
#
#   The contents of this file are subject to the terms and conditions defined in the License.
#   You may not use, modify, or distribute this file except in compliance with the License.
#
#   SPDX-License-Identifier: Apache-2.0
#
from deap_er.records.dtypes import *
from deap_er.records import Logbook
from deap_er.base import Toolbox
from .variation import *


__all__ = ["ea_mu_comma_lambda"]


def ea_mu_comma_lambda(
    toolbox: Toolbox,
    population: list,
    generations: int,
    offsprings: int,
    survivors: int,
    cx_prob: float,
    mut_prob: float,
    hof: Hof = None,
    stats: Stats = None,
    verbose: bool = False,
) -> AlgoResult:
    """Evolve a population with mu-comma-lambda selection.

    Requires ``mate``, ``mutate``, ``select``, and ``evaluate`` on
    ``toolbox``. Survivors are selected from the offspring only.

    Args:
        toolbox: Toolbox with the evolution operators.
        population: Individuals to evolve. Replaced in place.
        generations: Number of generations to run.
        offsprings: Number of offspring to produce each generation.
        survivors: Number of individuals to keep after selection.
        cx_prob: Probability of mating two individuals.
        mut_prob: Probability of mutating an individual.
        hof: Optional HallOfFame or ParetoFront to update.
        stats: Optional Statistics or MultiStatistics to compile.
        verbose: If True, print the logbook stream each generation.

    Returns:
        The final population and the logbook.
    """
    if survivors > offsprings:  # pragma: no cover
        offsprings, survivors = survivors, offsprings

    logbook = Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

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

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
from deap_er.base import Toolbox
from deap_er.records import Logbook
from deap_er.records.dtypes import *

from .variation import *

__all__ = ["ea_simple"]


def ea_simple(
    toolbox: Toolbox,
    population: list[Individual],
    generations: int,
    cx_prob: float,
    mut_prob: float,
    hof: Hof | None = None,
    stats: Stats | None = None,
    verbose: bool = False,
) -> AlgoResult:
    """Evolve a population with crossover and mutation on every generation.

    Requires ``mate``, ``mutate``, ``select``, and ``evaluate`` on
    ``toolbox``. Survivors are the offspring of the current generation.

    Args:
        toolbox: Toolbox with the evolution operators.
        population: Individuals to evolve. Replaced in place.
        generations: Number of generations to run.
        cx_prob: Probability of mating two individuals.
        mut_prob: Probability of mutating an individual.
        hof: Optional HallOfFame or ParetoFront to update.
        stats: Optional Statistics or MultiStatistics to compile.
        verbose: If True, print the logbook stream each generation.

    Returns:
        The final population and the logbook.
    """
    logbook = Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    for gen in range(1, generations + 1):
        offspring = toolbox.select(population, len(population))
        offspring = var_and(toolbox, offspring, cx_prob, mut_prob)

        invalids = [ind for ind in offspring if not ind.fitness.is_valid()]
        fitness = toolbox.map(toolbox.evaluate, invalids)
        for ind, fit in zip(invalids, fitness, strict=False):
            ind.fitness.values = fit

        population[:] = offspring

        if hof is not None:
            hof.update(offspring)
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalids), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

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


__all__ = ["ea_generate_update"]


def ea_generate_update(
    toolbox: Toolbox,
    generations: int,
    hof: Hof | None = None,
    stats: Stats | None = None,
    verbose: bool = False,
) -> AlgoResult:
    """Evolve a strategy that generates and updates a population.

    Requires ``generate``, ``update``, and ``evaluate`` on ``toolbox``.

    Args:
        toolbox: Toolbox with the generate, update, and evaluate operators.
        generations: Number of generations to run.
        hof: Optional HallOfFame or ParetoFront to update.
        stats: Optional Statistics or MultiStatistics to compile.
        verbose: If True, print the logbook stream each generation.

    Returns:
        The final population and the logbook.
    """
    logbook = Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

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

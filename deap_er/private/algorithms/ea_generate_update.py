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
from deap_er.private.toolbox import Toolbox
from deap_er.private.typedefs import EvoAlgoResult, EvoRecords, EvoStats, Individual

from .loop import new_logbook, record_generation

__all__ = ["ea_generate_update"]


def ea_generate_update(
    toolbox: Toolbox,
    generations: int,
    hof: EvoRecords | None = None,
    stats: EvoStats | None = None,
    verbose: bool = False,
) -> EvoAlgoResult:
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
    logbook = new_logbook(stats)

    population: list[Individual] = []
    for gen in range(1, generations + 1):
        population = toolbox.generate()

        fitness = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitness, strict=False):
            ind.fitness.values = fit

        toolbox.update(population)

        record_generation(
            logbook,
            gen,
            len(population),
            population=population,
            offspring=population,
            hof=hof,
            stats=stats,
            verbose=verbose,
        )

    return population, logbook

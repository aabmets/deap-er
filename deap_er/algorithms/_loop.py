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
from deap_er.records.typedefs import Hof, Individual, Stats

__all__: list[str] = []


def _new_logbook(stats: Stats | None) -> Logbook:
    """Create a logbook with the standard algorithm header.

    Args:
        stats: Optional Statistics or MultiStatistics whose fields
            become the trailing header columns.

    Returns:
        A logbook ready to record generations.
    """
    logbook = Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
    return logbook


def _evaluate_invalid(toolbox: Toolbox, individuals: list[Individual]) -> int:
    """Evaluate the individuals whose fitness is invalid.

    Args:
        toolbox: Toolbox with the evaluate and map operators.
        individuals: Individuals to scan for invalid fitness.

    Returns:
        The number of individuals that were evaluated.
    """
    invalids = [ind for ind in individuals if not ind.fitness.is_valid()]
    fitness = toolbox.map(toolbox.evaluate, invalids)
    for ind, fit in zip(invalids, fitness, strict=False):
        ind.fitness.values = fit
    return len(invalids)


def _record_generation(
    logbook: Logbook,
    gen: int,
    nevals: int,
    *,
    population: list[Individual],
    offspring: list[Individual],
    hof: Hof | None,
    stats: Stats | None,
    verbose: bool,
) -> None:
    """Update the hall of fame and append one generation to the logbook.

    The hall of fame sees ``offspring``, while statistics are compiled
    from ``population``. The two differ whenever survivors are chosen
    from a pool larger than the offspring.

    Args:
        logbook: Logbook to append the record to.
        gen: Generation number of the record.
        nevals: Number of individuals evaluated this generation.
        population: Individuals to compile statistics from.
        offspring: Individuals to offer to the hall of fame.
        hof: Optional HallOfFame or ParetoFront to update.
        stats: Optional Statistics or MultiStatistics to compile.
        verbose: If True, print the logbook stream.
    """
    if hof is not None:
        hof.update(offspring)
    record = stats.compile(population) if stats else {}
    logbook.record(gen=gen, nevals=nevals, **record)
    if verbose:
        print(logbook.stream)

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
from collections.abc import Sequence
from typing import Any

from deap_er.base import Toolbox
from deap_er.records import Logbook
from deap_er.records.typedefs import Hof, Individual, Stats

__all__: list[str] = ["new_logbook", "evaluate_invalid", "record_generation"]


def new_logbook(stats: Stats | None) -> Logbook:
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


def evaluate_invalid(toolbox: Toolbox, individuals: Sequence[Any]) -> int:
    """Evaluate the individuals whose fitness is invalid.

    When the toolbox has an ``evaluate_batch`` operator, the whole
    batch of invalid individuals is handed to it in one call and
    ``map`` is not used. Otherwise each individual goes through
    ``map`` and ``evaluate`` as usual.

    Args:
        toolbox: Toolbox with the evaluate and map operators.
        individuals: Individuals to scan for invalid fitness.

    Returns:
        The number of individuals that were evaluated.
    """
    invalids = [ind for ind in individuals if not ind.fitness.is_valid()]
    evaluate_batch = getattr(toolbox, "evaluate_batch", None)
    if evaluate_batch is not None:
        fitness = evaluate_batch(invalids)
    else:
        fitness = toolbox.map(toolbox.evaluate, invalids)
    for ind, fit in zip(invalids, fitness, strict=False):
        ind.fitness.values = fit
    return len(invalids)


def record_generation(
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

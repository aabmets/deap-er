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
import time
from collections.abc import Callable
from functools import partial
from logging import Logger
from typing import Any

from deap_er.private.toolbox import Toolbox
from deap_er.private.typedefs import EvoAlgoResult, EvoRecords, EvoStats, Individual

from .loop import budget_spent, check_n_evals, consume_evals, new_logbook, record_generation

__all__: list[str] = ["evolve_generations"]


def evolve_generations(
    toolbox: Toolbox,  # NOSONAR python:S107  mirrors the ea_* driver keywords
    population: list[Individual],
    generations: int,
    vary: Callable[[list[Individual]], list[Individual]],
    survive: Callable[[list[Individual], list[Individual]], list[Individual]],
    *,
    hof: EvoRecords | None,
    stats: EvoStats | None,
    verbose: bool,
    logger: Logger | None,
    log_time: bool,
    fronts: list[Any] | None,
    n_evals: int | None,
) -> EvoAlgoResult:
    """Run the shared generational loop of the population ``ea_*`` drivers.

    Generation zero evaluates ``population``. Every later generation
    builds offspring with ``vary``, evaluates them, and replaces the
    population in place with ``survive(population, offspring)``. The
    generation that meets or exceeds ``n_evals`` is the last one.

    Args:
        toolbox: Toolbox with the evaluate and map operators.
        population: Individuals to evolve. Replaced in place.
        generations: Number of generations after generation zero.
        vary: Maps the current population to unevaluated offspring.
        survive: Picks the next population from the current
            population and the evaluated offspring.
        hof: Optional HallOfFame or ParetoFront to update.
        stats: Optional Statistics or MultiStatistics to compile.
        verbose: If True, emit the logbook stream each generation.
        logger: If given with ``verbose``, the stream is logged.
        log_time: If True, record per-generation ``duration``.
        fronts: Optional list that receives a ParetoFront snapshot
            of each generation's population.
        n_evals: Optional evaluation budget.

    Returns:
        The final population and the logbook.

    Raises:
        ValueError: If ``n_evals`` is negative.
    """
    check_n_evals(n_evals)
    logbook = new_logbook(stats, log_time=log_time)
    record = partial(
        record_generation,
        logbook,
        hof=hof,
        stats=stats,
        verbose=verbose,
        logger=logger,
        fronts=fronts,
    )

    t0 = time.perf_counter()
    nevals, used = consume_evals(toolbox, population, 0)
    duration = time.perf_counter() - t0 if log_time else None
    record(0, nevals, population=population, offspring=population, duration=duration)
    if budget_spent(n_evals, used):
        return population, logbook

    for gen in range(1, generations + 1):
        t0 = time.perf_counter()
        offspring = vary(population)
        nevals, used = consume_evals(toolbox, offspring, used)
        population[:] = survive(population, offspring)
        duration = time.perf_counter() - t0 if log_time else None
        record(gen, nevals, population=population, offspring=offspring, duration=duration)
        if budget_spent(n_evals, used):
            break
    return population, logbook

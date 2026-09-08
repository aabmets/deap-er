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
from logging import Logger
from typing import Any

from deap_er.private.toolbox import Toolbox
from deap_er.private.typedefs import EvoAlgoResult, EvoRecords, EvoStats, Individual

from .loop import budget_spent, check_n_evals, consume_evals, new_logbook, record_generation
from .variation import var_and

__all__: list[str] = ["ea_simple"]


def ea_simple(
    toolbox: Toolbox,
    population: list[Individual],
    generations: int,
    cx_prob: float,
    mut_prob: float,
    hof: EvoRecords | None = None,
    stats: EvoStats | None = None,
    verbose: bool = False,
    logger: Logger | None = None,
    log_time: bool = False,
    fronts: list[Any] | None = None,
    n_evals: int | None = None,
) -> EvoAlgoResult:
    """Evolve a population with crossover and mutation on every generation.

    Requires ``mate``, ``mutate``, ``select``, and ``evaluate`` on
    ``toolbox``. Survivors are the offspring of the current generation.
    When ``n_evals`` is set, the generation that meets or exceeds that
    count is the last one recorded. Generations remain the default stop.

    Args:
        toolbox: Toolbox with the evolution operators.
        population: Individuals to evolve. Replaced in place.
        generations: Number of generations to run.
        cx_prob: Probability of mating two individuals.
        mut_prob: Probability of mutating an individual.
        hof: Optional HallOfFame or ParetoFront to update.
        stats: Optional Statistics or MultiStatistics to compile.
        verbose: If True, print the logbook stream each generation.
        logger: If given with ``verbose``, the stream is logged.
        log_time: If True, record per-generation ``duration``.
        fronts: Optional list that receives a ParetoFront snapshot
            of each generation's population.
        n_evals: Optional evaluation budget. The generation that
            meets or exceeds this count is finished, then the loop
            stops. ``None`` keeps the generation limit only. Counts
            fitness assignments through ``evaluate_invalid``,
            including ``EvalCache`` hits.

    Returns:
        The final population and the logbook.

    Raises:
        ValueError: If ``n_evals`` is negative.
    """
    check_n_evals(n_evals)
    logbook = new_logbook(stats, log_time=log_time)
    t0 = time.perf_counter()
    nevals, used = consume_evals(toolbox, population, n_evals, 0)
    duration = time.perf_counter() - t0 if log_time else None
    record_generation(
        logbook,
        0,
        nevals,
        population=population,
        offspring=population,
        hof=hof,
        stats=stats,
        verbose=verbose,
        logger=logger,
        duration=duration,
        fronts=fronts,
    )
    if budget_spent(n_evals, used):
        return population, logbook

    for gen in range(1, generations + 1):
        t0 = time.perf_counter()
        offspring = toolbox.select(population, len(population))
        offspring = var_and(toolbox, offspring, cx_prob, mut_prob)

        nevals, used = consume_evals(toolbox, offspring, n_evals, used)

        population[:] = offspring
        duration = time.perf_counter() - t0 if log_time else None

        record_generation(
            logbook,
            gen,
            nevals,
            population=population,
            offspring=offspring,
            hof=hof,
            stats=stats,
            verbose=verbose,
            logger=logger,
            duration=duration,
            fronts=fronts,
        )
        if budget_spent(n_evals, used):
            break

    return population, logbook

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

from .loop import evaluate_invalid, new_logbook, record_generation
from .variation import var_or

__all__ = ["ea_mu_comma_lambda"]


def ea_mu_comma_lambda(
    toolbox: Toolbox,
    population: list[Individual],
    generations: int,
    offsprings: int,
    survivors: int,
    cx_prob: float,
    mut_prob: float,
    hof: EvoRecords | None = None,
    stats: EvoStats | None = None,
    verbose: bool = False,
    logger: Logger | None = None,
    log_time: bool = False,
    fronts: list[Any] | None = None,
) -> EvoAlgoResult:
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
        logger: If given with ``verbose``, the stream is logged.
        log_time: If True, record per-generation ``duration``.
        fronts: Optional list that receives a ParetoFront snapshot
            of each generation's population.

    Returns:
        The final population and the logbook.

    Raises:
        ValueError: If ``survivors`` is greater than ``offsprings``.
    """
    if survivors > offsprings:
        raise ValueError(
            "The number of survivors must be less than or equal to the number of offsprings."
        )

    logbook = new_logbook(stats, log_time=log_time)
    t0 = time.perf_counter()
    nevals = evaluate_invalid(toolbox, population)
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

    for gen in range(1, generations + 1):
        t0 = time.perf_counter()
        offspring = var_or(toolbox, population, offsprings, cx_prob, mut_prob)

        nevals = evaluate_invalid(toolbox, offspring)

        population[:] = toolbox.select(offspring, survivors)
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

    return population, logbook

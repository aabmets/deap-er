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

from deap_er.private.algorithms.loop import new_logbook, record_generation
from deap_er.private.strategies.restart import RestartStrategy
from deap_er.private.toolbox import Toolbox
from deap_er.private.typedefs import EvoAlgoResult, EvoRecords, EvoStats, Individual

__all__ = ["ea_generate_update_restarts"]


def ea_generate_update_restarts(
    toolbox: Toolbox,
    restart_strategy: RestartStrategy,
    hof: EvoRecords | None = None,
    stats: EvoStats | None = None,
    verbose: bool = False,
    logger: Logger | None = None,
    log_time: bool = False,
    log_restarts: bool = True,
    fronts: list[Any] | None = None,
) -> EvoAlgoResult:
    """Evolve with IPOP or BIPOP CMA restarts until the budget is spent.

    Requires ``generate``, ``update``, and ``evaluate`` on ``toolbox``.
    The toolbox operators should be bound to ``restart_strategy.generate``
    and ``restart_strategy.update``.

    Args:
        toolbox: Toolbox with generate, update, and evaluate operators.
        restart_strategy: Restart wrapper that tracks stagnation and
            relaunches the inner CMA strategy.
        hof: Optional HallOfFame or ParetoFront to update.
        stats: Optional Statistics or MultiStatistics to compile.
        verbose: If True, print the logbook stream each generation.
        logger: If given with ``verbose``, the stream is logged.
        log_time: If True, record per-generation ``duration``.
        log_restarts: If True, log ``restart``, ``regime``, ``lambda``,
            and ``evals`` columns.
        fronts: Optional list that receives a ParetoFront snapshot
            of each generation's population.

    Returns:
        The final population and the logbook.
    """
    logbook = new_logbook(stats, log_time=log_time)
    if log_restarts:
        logbook.header.extend(["restart", "regime", "lambda", "evals"])

    population: list[Individual] = []
    gen = 0
    while not restart_strategy.is_done():
        t0 = time.perf_counter()
        population = toolbox.generate()
        fitness = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitness, strict=False):
            ind.fitness.values = fit

        restart_strategy.update(population)
        gen += 1
        duration = time.perf_counter() - t0 if log_time else None

        extra: dict[str, Any] = {}
        if log_restarts:
            inner = restart_strategy.strategy
            extra = {
                "restart": restart_strategy.restart_count,
                "regime": restart_strategy.regime or "initial",
                "lambda": getattr(inner, "lamb", None),
                "evals": restart_strategy.evals_used,
            }

        record_generation(
            logbook,
            gen,
            len(population),
            population=population,
            offspring=population,
            hof=hof,
            stats=stats,
            verbose=False,
            logger=logger,
            duration=duration,
            fronts=fronts,
        )
        if extra:
            logbook[-1].update(extra)
        if verbose:
            text = logbook.stream
            if logger is not None:
                logger.info(text)
            else:
                print(text)

        if restart_strategy.is_done():
            break
        if restart_strategy.should_restart():
            restart_strategy.restart()

    return population, logbook

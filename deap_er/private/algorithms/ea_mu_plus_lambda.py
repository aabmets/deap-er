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
from logging import Logger
from typing import Any

from deap_er.private.toolbox import Toolbox
from deap_er.private.typedefs import EvoAlgoResult, EvoRecords, EvoStats, Individual

from .generations import evolve_generations
from .variation import var_or

__all__ = ["ea_mu_plus_lambda"]


def ea_mu_plus_lambda(
    toolbox: Toolbox,  # NOSONAR python:S107  n_evals matches sibling ea_* drivers
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
    n_evals: int | None = None,
) -> EvoAlgoResult:
    """Evolve a population with mu-plus-lambda selection.

    Requires ``mate``, ``mutate``, ``select``, and ``evaluate`` on
    ``toolbox``. Survivors are selected from the union of parents
    and offspring. When ``n_evals`` is set, the generation that
    meets or exceeds that count is the last one recorded.
    Generations remain the default stop.

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
    return evolve_generations(
        toolbox,
        population,
        generations,
        lambda pop: var_or(toolbox, pop, offsprings, cx_prob, mut_prob),
        lambda pop, offspring: toolbox.select(pop + offspring, survivors),
        hof=hof,
        stats=stats,
        verbose=verbose,
        logger=logger,
        log_time=log_time,
        fronts=fronts,
        n_evals=n_evals,
    )

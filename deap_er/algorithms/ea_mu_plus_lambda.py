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
from deap_er.records.typedefs import AlgoResult, Hof, Individual, Stats

from ._loop import _evaluate_invalid, _new_logbook, _record_generation
from .variation import var_or

__all__ = ["ea_mu_plus_lambda"]


def ea_mu_plus_lambda(
    toolbox: Toolbox,
    population: list[Individual],
    generations: int,
    offsprings: int,
    survivors: int,
    cx_prob: float,
    mut_prob: float,
    hof: Hof | None = None,
    stats: Stats | None = None,
    verbose: bool = False,
) -> AlgoResult:
    """Evolve a population with mu-plus-lambda selection.

    Requires ``mate``, ``mutate``, ``select``, and ``evaluate`` on
    ``toolbox``. Survivors are selected from the union of parents
    and offspring.

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

    Returns:
        The final population and the logbook.
    """
    logbook = _new_logbook(stats)
    nevals = _evaluate_invalid(toolbox, population)
    _record_generation(
        logbook,
        0,
        nevals,
        population=population,
        offspring=population,
        hof=hof,
        stats=stats,
        verbose=verbose,
    )

    for gen in range(1, generations + 1):
        offspring = var_or(toolbox, population, offsprings, cx_prob, mut_prob)

        nevals = _evaluate_invalid(toolbox, offspring)

        population[:] = toolbox.select(population + offspring, survivors)

        _record_generation(
            logbook,
            gen,
            nevals,
            population=population,
            offspring=offspring,
            hof=hof,
            stats=stats,
            verbose=verbose,
        )

    return population, logbook

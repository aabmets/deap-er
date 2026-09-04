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
from deap_er.base.dtypes import Individual
import random


__all__ = ["var_and", "var_or"]


def var_and(
    toolbox: Toolbox, population: list[Individual], cx_prob: float, mut_prob: float
) -> list[Individual]:
    """Clone a population, then apply crossover and mutation independently.

    Each of ``cx_prob`` and ``mut_prob`` must be in ``[0, 1]``. The
    result is a new list; fitnesses of varied individuals are cleared.

    Requires ``clone``, ``mate``, and ``mutate`` on ``toolbox``.

    Args:
        toolbox: Toolbox with the variation operators.
        population: Individuals to vary.
        cx_prob: Probability of mating each consecutive pair.
        mut_prob: Probability of mutating each individual.

    Returns:
        A new list of varied individuals.

    Raises:
        ValueError: If either probability is outside ``[0, 1]``.
    """
    err = "The {0} probability must be in the range of [0, 1]."
    if not (0 <= cx_prob <= 1):
        raise ValueError(err.format("crossover"))
    if not (0 <= mut_prob <= 1):
        raise ValueError(err.format("mutation"))

    offspring = [toolbox.clone(ind) for ind in population]

    for i in range(1, len(offspring), 2):
        if random.random() < cx_prob:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mut_prob:
            (offspring[i],) = toolbox.mutate(offspring[i])  # don't remove the comma!
            del offspring[i].fitness.values

    return offspring


def var_or(
    toolbox: Toolbox,
    population: list[Individual],
    offsprings: int,
    cx_prob: float,
    mut_prob: float,
) -> list[Individual]:
    """Build offspring by applying crossover *or* mutation *or* copy.

    The sum of ``cx_prob`` and ``mut_prob`` must be in ``[0, 1]``. The
    remaining probability copies an unmodified parent. The result is a
    new list; fitnesses of varied individuals are cleared.

    Requires ``clone``, ``mate``, and ``mutate`` on ``toolbox``.

    Args:
        toolbox: Toolbox with the variation operators.
        population: Individuals to sample from.
        offsprings: Number of individuals to produce.
        cx_prob: Probability of producing a child by crossover.
        mut_prob: Probability of producing a child by mutation.

    Returns:
        A new list of offspring.

    Raises:
        ValueError: If ``cx_prob + mut_prob`` is greater than 1.
    """
    evolve_prob = cx_prob + mut_prob
    if evolve_prob > 1.0:
        raise ValueError(
            "The sum of the crossover and the mutation "
            "probabilities must be in the range of [0, 1]."
        )

    offspring = []
    for _ in range(offsprings):
        op_choice = random.random()
        if op_choice < cx_prob:
            ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < evolve_prob:
            ind = toolbox.clone(random.choice(population))
            (ind,) = toolbox.mutate(ind)  # don't remove the comma!
            del ind.fitness.values
            offspring.append(ind)
        else:
            offspring.append(random.choice(population))

    return offspring

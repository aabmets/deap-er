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

import pytest
from deap_er import Fitness, Toolbox, creator, tools

VAR_FIT = "VAR_FIT"
VAR_IND = "VAR_IND"


@pytest.fixture
def toolbox():
    creator.create_type(VAR_FIT, Fitness, weights=(-1.0,))
    creator.create_type(VAR_IND, list, fitness=creator.__dict__[VAR_FIT])

    tb = Toolbox()
    tb.register("mate", tools.cx_two_point)
    tb.register("mutate", tools.mut_flip_bit, mut_prob=0.5)

    yield tb

    del creator.__dict__[VAR_FIT]
    del creator.__dict__[VAR_IND]


def _population(count=6):
    population = []
    for i in range(count):
        ind = creator.__dict__[VAR_IND]([i % 2] * 4)
        ind.fitness.values = (float(i),)
        population.append(ind)
    return population


def test_var_or_never_returns_a_parent_object(toolbox):
    # Reproduction must hand back an independent copy, so that later
    # mutation of an offspring cannot reach back into the population.
    tools.rng.seed(5)
    population = _population()

    offspring = tools.var_or(toolbox, population, 40, cx_prob=0.0, mut_prob=0.0)

    assert len(offspring) == 40
    parents = {id(ind) for ind in population}
    assert all(id(child) not in parents for child in offspring)


def test_var_or_reproduction_copies_the_genes(toolbox):
    tools.rng.seed(5)
    population = _population()

    offspring = tools.var_or(toolbox, population, 10, cx_prob=0.0, mut_prob=0.0)

    assert all(list(child) in [list(p) for p in population] for child in offspring)


def test_var_or_mutating_offspring_leaves_parents_untouched(toolbox):
    tools.rng.seed(7)
    population = _population()
    before = [list(ind) for ind in population]

    offspring = tools.var_or(toolbox, population, 20, cx_prob=0.0, mut_prob=0.0)
    for child in offspring:
        child[0] = 99

    assert [list(ind) for ind in population] == before


def test_var_or_single_parent_with_crossover_does_not_crash(toolbox):
    population = _population(count=1)

    offspring = tools.var_or(toolbox, population, 6, cx_prob=1.0, mut_prob=0.0)

    assert len(offspring) == 6
    parent_id = id(population[0])
    assert all(id(child) != parent_id for child in offspring)


def test_var_or_still_produces_the_requested_count(toolbox):
    tools.rng.seed(9)
    population = _population()

    offspring = tools.var_or(toolbox, population, 12, cx_prob=0.5, mut_prob=0.3)

    assert len(offspring) == 12

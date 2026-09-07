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

ISLAND_FIT = "ISLAND_FIT"
ISLAND_IND = "ISLAND_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(ISLAND_FIT, Fitness, weights=(1.0,))
    creator.create_type(ISLAND_IND, list, fitness=creator.__dict__[ISLAND_FIT])
    yield creator.__dict__[ISLAND_IND]
    del creator.__dict__[ISLAND_FIT]
    del creator.__dict__[ISLAND_IND]


def _evaluate(individual):
    return (float(individual[0]),)


def _individuals(ind_cls, values):
    population = []
    for value in values:
        ind = ind_cls([value])
        population.append(ind)
    return population


def _toolbox(vary, select, evaluate=_evaluate):
    toolbox = Toolbox()
    toolbox.register("evaluate", evaluate)
    toolbox.register("vary", vary)
    toolbox.register("select", select)
    return toolbox


def _vary_plus(ind_cls, extras):
    def vary(population):
        added = _individuals(ind_cls, extras)
        return list(population) + added

    return vary


def test_step_islands_applies_different_select_pressures(ind_cls):
    low = _individuals(ind_cls, [0, 1, 2])
    high = _individuals(ind_cls, [0, 1, 2])
    best = _toolbox(_vary_plus(ind_cls, [10, 11, 12]), tools.sel_best)
    worst = _toolbox(_vary_plus(ind_cls, [10, 11, 12]), tools.sel_worst)

    tools.step_islands([(best, low), (worst, high)])

    assert sorted(ind[0] for ind in low) == [10, 11, 12]
    assert sorted(ind[0] for ind in high) == [0, 1, 2]


def test_step_islands_evaluates_offspring_before_select(ind_cls):
    seen: list[bool] = []

    def select(individuals, sel_count):
        seen.append(all(ind.fitness.is_valid() for ind in individuals))
        return list(individuals[:sel_count])

    population = _individuals(ind_cls, [1, 2])
    toolbox = _toolbox(_vary_plus(ind_cls, [9, 8]), select)

    tools.step_islands([(toolbox, population)])

    assert seen == [True]
    assert all(ind.fitness.is_valid() for ind in population)


def test_step_islands_uses_evaluate_batch_when_registered(ind_cls):
    def evaluate_batch(individuals):
        return [_evaluate(ind) for ind in individuals]

    def explode(_individual):
        raise AssertionError("map must not be used while evaluate_batch is registered")

    toolbox = Toolbox()
    toolbox.register("evaluate", explode)
    toolbox.register("evaluate_batch", evaluate_batch)
    toolbox.register("vary", _vary_plus(ind_cls, [4]))
    toolbox.register("select", tools.sel_best)
    toolbox.register("map", explode)
    population = _individuals(ind_cls, [1, 2, 3])

    tools.step_islands([(toolbox, population)])

    assert sorted(ind[0] for ind in population) == [2, 3, 4]


def test_step_islands_without_migrate_leaves_membership(ind_cls):
    first = _individuals(ind_cls, [1, 2])
    second = _individuals(ind_cls, [3, 4])
    keep = _toolbox(list, tools.sel_best)

    tools.step_islands([(keep, first), (keep, second)])

    assert sorted(ind[0] for ind in first) == [1, 2]
    assert sorted(ind[0] for ind in second) == [3, 4]


def test_step_islands_migrate_moves_individuals(ind_cls):
    first = _individuals(ind_cls, [1, 2])
    second = _individuals(ind_cls, [8, 9])
    for ind in first + second:
        ind.fitness.values = _evaluate(ind)
    keep = _toolbox(list, tools.sel_best)

    tools.step_islands(
        [(keep, first), (keep, second)],
        migrate=lambda pops: tools.mig_ring(pops, 1, tools.sel_best),
    )

    assert [len(first), len(second)] == [2, 2]
    assert any(ind[0] in {8, 9} for ind in first)
    assert any(ind[0] in {1, 2} for ind in second)


def test_step_islands_keeps_fitness_when_eval_keys_match(ind_cls):
    first = _individuals(ind_cls, [1, 2])
    second = _individuals(ind_cls, [8, 9])
    for ind in first + second:
        ind.fitness.values = _evaluate(ind)
    keep = _toolbox(list, tools.sel_best)

    tools.step_islands(
        [(keep, first), (keep, second)],
        migrate=lambda pops: tools.mig_ring(pops, 1, tools.sel_best),
        eval_keys=("shared", "shared"),
    )

    assert all(ind.fitness.is_valid() for ind in first + second)


def test_step_islands_invalidates_immigrants_when_eval_keys_differ(ind_cls):
    first = _individuals(ind_cls, [1, 2])
    second = _individuals(ind_cls, [8, 9])
    for ind in first + second:
        ind.fitness.values = _evaluate(ind)
    first_ids = {id(ind) for ind in first}
    keep = _toolbox(list, tools.sel_best)

    tools.step_islands(
        [(keep, first), (keep, second)],
        migrate=lambda pops: tools.mig_ring(pops, 1, tools.sel_best),
        eval_keys=("cases-a", "cases-b"),
    )

    natives = [ind for ind in first if id(ind) in first_ids]
    arrivals = [ind for ind in first if id(ind) not in first_ids]
    assert arrivals
    assert all(ind.fitness.is_valid() for ind in natives)
    assert all(not ind.fitness.is_valid() for ind in arrivals)


def test_step_islands_invalidates_replacement_clones(ind_cls):
    first = _individuals(ind_cls, [1, 2, 3])
    second = _individuals(ind_cls, [8, 9, 10])
    for ind in first + second:
        ind.fitness.values = _evaluate(ind)
    second_ids = {id(ind) for ind in second}
    keep = _toolbox(list, tools.sel_best)

    tools.step_islands(
        [(keep, first), (keep, second)],
        migrate=lambda pops: tools.mig_ring(pops, 1, tools.sel_best, replacement=tools.sel_worst),
        eval_keys=("matrix-a", "matrix-b"),
    )

    clones = [ind for ind in second if id(ind) not in second_ids]
    assert clones
    assert all(not ind.fitness.is_valid() for ind in clones)


def test_step_islands_rejects_a_toolbox_without_vary(ind_cls):
    toolbox = Toolbox()
    toolbox.register("evaluate", _evaluate)
    toolbox.register("select", tools.sel_best)
    population = _individuals(ind_cls, [1])

    with pytest.raises(ValueError, match="vary"):
        tools.step_islands([(toolbox, population)])


def test_step_islands_rejects_a_toolbox_without_select(ind_cls):
    toolbox = Toolbox()
    toolbox.register("evaluate", _evaluate)
    toolbox.register("vary", list)
    population = _individuals(ind_cls, [1])

    with pytest.raises(ValueError, match="select"):
        tools.step_islands([(toolbox, population)])


def test_step_islands_rejects_eval_keys_of_the_wrong_length(ind_cls):
    toolbox = _toolbox(list, tools.sel_best)
    population = _individuals(ind_cls, [1])

    with pytest.raises(ValueError, match="eval_keys"):
        tools.step_islands([(toolbox, population)], eval_keys=("a", "b"))


def test_step_islands_empty_demes_is_a_noop():
    tools.step_islands([])

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
import operator
import random
from functools import partial

import pytest
from deap_er import base, creator, gp, tools

HARM_FIT = "HARM_FIT"
HARM_IND = "HARM_IND"


def _evaluate(individual, toolbox):
    func = toolbox.compile(expr=individual)
    return (sum(abs(func(x) - x * x) for x in range(-3, 4)),)


@pytest.fixture
def toolbox():
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.sub, 2)
    pset.add_primitive(operator.mul, 2)

    creator.create(HARM_FIT, base.Fitness, weights=(-1.0,))
    creator.create(HARM_IND, gp.PrimitiveTree, fitness=creator.__dict__[HARM_FIT])

    expr = partial(gp.gen_half_and_half, prim_set=pset, min_depth=1, max_depth=2)
    expr_mut = partial(gp.gen_full, min_depth=0, max_depth=2)

    tb = base.Toolbox()
    tb.register("individual", tools.init_iterate, creator.__dict__[HARM_IND], expr)
    tb.register("population", tools.init_repeat, list, tb.individual)
    tb.register("compile", gp.compile_tree, prim_set=pset)
    tb.register("mate", gp.cx_one_point)
    tb.register("mutate", gp.mut_uniform, expr=expr_mut, prim_set=pset)
    tb.register("select", tools.sel_tournament, contestants=3)
    tb.register("evaluate", _evaluate, toolbox=tb)

    yield tb

    del creator.__dict__[HARM_FIT]
    del creator.__dict__[HARM_IND]


def _seeded_population(toolbox):
    random.seed(31)
    return toolbox.population(size=20)


def test_harm_records_generation_zero_then_each_generation(toolbox):
    population = _seeded_population(toolbox)

    _, logbook = gp.harm(
        toolbox=toolbox,
        population=population,
        generations=3,
        cx_prob=0.5,
        mut_prob=0.1,
        nb_model=40,
    )

    assert logbook.select("gen") == [0, 1, 2, 3]


def test_harm_evaluation_counts_are_stable(toolbox):
    # Pins the RNG consumption order of the offspring-production loop.
    population = _seeded_population(toolbox)

    _, logbook = gp.harm(
        toolbox=toolbox,
        population=population,
        generations=3,
        cx_prob=0.5,
        mut_prob=0.1,
        nb_model=40,
    )

    assert logbook.select("nevals") == [20, 12, 15, 16]


def test_harm_replaces_the_population_in_place(toolbox):
    population = _seeded_population(toolbox)

    final, _ = gp.harm(
        toolbox=toolbox,
        population=population,
        generations=3,
        cx_prob=0.5,
        mut_prob=0.1,
        nb_model=40,
    )

    assert final is population
    assert len(final) == 20
    assert all(ind.fitness.is_valid() for ind in final)


def test_harm_updates_the_hall_of_fame(toolbox):
    population = _seeded_population(toolbox)
    hof = tools.HallOfFame(1)

    gp.harm(
        toolbox=toolbox,
        population=population,
        generations=3,
        cx_prob=0.5,
        mut_prob=0.1,
        nb_model=40,
        hof=hof,
    )

    assert len(hof) == 1
    assert hof[0].fitness.values == (0.0,)


def test_harm_compiles_statistics_into_the_logbook(toolbox):
    population = _seeded_population(toolbox)
    stats = tools.Statistics(len)
    stats.register("max", max)

    _, logbook = gp.harm(
        toolbox=toolbox,
        population=population,
        generations=3,
        cx_prob=0.5,
        mut_prob=0.1,
        nb_model=40,
        stats=stats,
    )

    assert logbook.select("max") == [7, 7, 7, 7]

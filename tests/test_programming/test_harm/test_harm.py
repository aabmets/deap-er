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
from functools import partial

import pytest
from deap_er import Fitness, Toolbox, creator, gp, tools

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

    creator.create_type(HARM_FIT, Fitness, weights=(-1.0,))
    creator.create_type(HARM_IND, gp.PrimitiveTree, fitness=creator.__dict__[HARM_FIT])

    expr = partial(gp.gen_half_and_half, prim_set=pset, min_depth=1, max_depth=2)
    expr_mut = partial(gp.gen_full, min_depth=0, max_depth=2)

    tb = Toolbox()
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
    tools.rng.seed(31)
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

    assert logbook.select("nevals") == [20, 15, 9, 14]


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


def test_target_prob_half_life_scales_with_cutoff_not_size():
    import math

    from deap_er.private.programming.harm.harm_size import target_prob

    alpha, beta, gamma, pop_len, cutoff = 0.05, 10.0, 0.25, 100, 20
    tau = cutoff * alpha + beta

    def expected(size: int) -> float:
        amplitude = gamma * pop_len * math.log(2) / tau
        decay = math.exp(-math.log(2) * (size - cutoff) / tau)
        return amplitude * decay

    assert target_prob(30, alpha, beta, gamma, pop_len, cutoff) == pytest.approx(expected(30))
    assert target_prob(40, alpha, beta, gamma, pop_len, cutoff) == pytest.approx(expected(40))


def test_harm_cutoff_uses_evaluated_population_and_survives_small_model(toolbox):
    population = toolbox.population(size=30)
    for individual in population:
        individual.fitness.values = toolbox.evaluate(individual)
    gp.harm(
        toolbox=toolbox,
        population=population,
        generations=1,
        cx_prob=0.5,
        mut_prob=0.1,
        nb_model=10,
        min_cutoff=1,
        rho=0.9,
    )


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

    assert logbook.select("max") == [7, 7, 7, 3]


def test_harm_rejects_unknown_kwargs(toolbox):
    population = _seeded_population(toolbox)
    with pytest.raises(TypeError, match="unexpected"):
        gp.harm(
            toolbox,
            population,
            generations=0,
            cx_prob=0.5,
            mut_prob=0.1,
            bogus=1,
        )


def test_harm_verbose_prints_and_default_model_size(toolbox, capsys):
    population = toolbox.population(size=8)
    gp.harm(
        toolbox,
        population,
        generations=0,
        cx_prob=0.5,
        mut_prob=0.1,
        verbose=True,
    )
    gp.harm(
        toolbox,
        population,
        generations=1,
        cx_prob=0.5,
        mut_prob=0.1,
        nb_model=20,
        verbose=True,
    )
    assert capsys.readouterr().out

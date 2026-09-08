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
from deap_er import Fitness, Toolbox, creator, tools

FIT = "AGE2_FIT"
IND = "AGE2_IND"
HV_THRESHOLD = 110.0


def test_empty_population_returns_empty():
    assert tools.sel_age_moea_2([], 3) == []


def test_partial_second_front_uses_geometry_scores():
    fit_name = "AGE2_F2_FIT"
    ind_name = "AGE2_F2_IND"
    creator.create_type(fit_name, Fitness, weights=(-1.0, -1.0))
    creator.create_type(ind_name, list, fitness=creator.__dict__[fit_name])
    try:
        front0 = []
        for genes, values in (([0.0], (0.0, 1.0)), ([1.0], (1.0, 0.0))):
            ind = creator.__dict__[ind_name](genes)
            ind.fitness.values = values
            front0.append(ind)

        front1 = []
        for genes, values in (([2.0], (0.6, 0.4)), ([3.0], (0.7, 0.35)), ([4.0], (0.8, 0.3))):
            ind = creator.__dict__[ind_name](genes)
            ind.fitness.values = values
            front1.append(ind)

        pop = front0 + front1
        chosen = tools.sel_age_moea_2(pop, 4)
        assert len(chosen) == 4
        assert {id(ind) for ind in chosen[:2]} == {id(front0[0]), id(front0[1])}
    finally:
        del creator.__dict__[fit_name]
        del creator.__dict__[ind_name]


def test_age_moea_2_zdt1():
    creator.create_type(FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        dimensions = 5
        bound_low, bound_up = 0.0, 1.0
        survivors = 16
        generations = 100

        toolbox = Toolbox()
        toolbox.register("attr_float", tools.rng.uniform, bound_low, bound_up)
        toolbox.register(
            "individual",
            tools.init_repeat,
            creator.__dict__[IND],
            toolbox.attr_float,
            dimensions,
        )
        toolbox.register("population", tools.init_repeat, list, toolbox.individual)
        toolbox.register(
            "mate", tools.cx_simulated_binary_bounded, low=bound_low, up=bound_up, eta=20.0
        )
        toolbox.register(
            "mutate",
            tools.mut_polynomial_bounded,
            low=bound_low,
            up=bound_up,
            eta=20.0,
            mut_prob=1.0 / dimensions,
        )
        toolbox.register("evaluate", tools.bm_zdt_1)
        toolbox.register("select", tools.sel_age_moea_2)

        pop = toolbox.population(size=survivors)
        fitness = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitness, strict=False):
            ind.fitness.values = fit

        pop = toolbox.select(pop, len(pop))
        for _ in range(1, generations):
            offspring = tools.var_and(toolbox, pop, 1.0, 1.0)
            invalid_ind = [ind for ind in offspring if not ind.fitness.is_valid()]
            fitness = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitness, strict=False):
                ind.fitness.values = fit
            pop = toolbox.select(pop + offspring, survivors)

        hv = tools.hypervolume(pop, [11.0, 11.0])
        assert hv > HV_THRESHOLD
    finally:
        del creator.__dict__[FIT]
        del creator.__dict__[IND]


def test_age_moea_2_dtlz2_three_objectives():
    creator.create_type(FIT, Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        dimensions = 7
        objectives = 3
        bound_low, bound_up = 0.0, 1.0
        survivors = 45
        generations = 50

        toolbox = Toolbox()
        toolbox.register("attr_float", tools.rng.uniform, bound_low, bound_up)
        toolbox.register(
            "individual",
            tools.init_repeat,
            creator.__dict__[IND],
            toolbox.attr_float,
            dimensions,
        )
        toolbox.register("population", tools.init_repeat, list, toolbox.individual)
        toolbox.register(
            "mate", tools.cx_simulated_binary_bounded, low=bound_low, up=bound_up, eta=20.0
        )
        toolbox.register(
            "mutate",
            tools.mut_polynomial_bounded,
            low=bound_low,
            up=bound_up,
            eta=20.0,
            mut_prob=1.0 / dimensions,
        )
        toolbox.register("evaluate", tools.bm_dtlz_2, count=objectives)
        toolbox.register("select", tools.SelAGE2WithMemory())

        pop = toolbox.population(size=survivors)
        fitness = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitness, strict=False):
            ind.fitness.values = tuple(fit)

        pop = toolbox.select(pop, len(pop))
        for _ in range(1, generations):
            offspring = tools.var_and(toolbox, pop, 1.0, 1.0)
            invalid_ind = [ind for ind in offspring if not ind.fitness.is_valid()]
            fitness = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitness, strict=False):
                ind.fitness.values = tuple(fit)
            pop = toolbox.select(pop + offspring, survivors)

        ref = [2.0, 2.0, 2.0]
        hv = tools.hypervolume(pop, ref)
        assert hv > 0.5
    finally:
        del creator.__dict__[FIT]
        del creator.__dict__[IND]

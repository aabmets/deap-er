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
import numpy
import pytest
from deap_er import Fitness, Toolbox, creator, tools

FIT = "MOEAD_FIT"
IND = "MOEAD_IND"
HV_THRESHOLD = 110.0


def test_empty_population_returns_empty():
    weights = tools.uniform_reference_points(2, 4)
    assert tools.sel_moead([], 3, weights) == []


def test_zero_sel_count_returns_empty(multi_obj, make):
    weights = tools.uniform_reference_points(2, 4)
    pop = [make(multi_obj, [0.0], (0.0, 1.0))]
    assert tools.sel_moead(pop, 0, weights) == []


def test_tchebycheff_known_values():
    fitness = numpy.array([[1.0, 4.0], [3.0, 2.0]])
    weights = numpy.array([[1.0, 0.0], [0.0, 1.0]])
    ideal = numpy.array([0.0, 0.0])
    got = tools.scalarization_tchebycheff(fitness, weights, ideal)
    assert got[0, 0] == pytest.approx(1.0)
    assert got[0, 1] == pytest.approx(4.0)
    assert got[1, 0] == pytest.approx(3.0)
    assert got[1, 1] == pytest.approx(2.0)


def test_pbi_known_values():
    fitness = numpy.array([[1.0, 1.0]])
    weights = numpy.array([[1.0, 1.0]])
    ideal = numpy.array([0.0, 0.0])
    got = tools.scalarization_pbi(fitness, weights, ideal, theta=5.0)
    d1 = numpy.sqrt(2.0)
    assert got[0, 0] == pytest.approx(d1)


def test_zero_weight_eps():
    fitness = numpy.array([[1.0, 2.0]])
    weights = numpy.array([[0.0, 1.0]])
    ideal = numpy.array([0.0, 0.0])
    got = tools.scalarization_tchebycheff(fitness, weights, ideal)
    assert numpy.isfinite(got).all()


def test_ideal_point_memory_updates(multi_obj, make):
    weights = tools.uniform_reference_points(2, 4)
    select = tools.SelMOEADWithMemory(weights)
    tools.rng.seed(3)
    for _ in range(2):
        pop = [
            make(multi_obj, [tools.rng.random()], (tools.rng.random(), tools.rng.random()))
            for _ in range(10)
        ]
        select(pop, 4)
    assert numpy.all(numpy.isfinite(select.ideal_point))
    assert numpy.all(select.ideal_point <= 1.0)


def test_subproblem_assignment():
    fit_name = "MOEAD_SUB_FIT"
    ind_name = "MOEAD_SUB_IND"
    creator.create_type(fit_name, Fitness, weights=(-1.0, -1.0))
    creator.create_type(ind_name, list, fitness=creator.__dict__[fit_name])
    try:
        pop = []
        for genes, values in (
            ([0.0], (0.0, 1.0)),
            ([1.0], (1.0, 0.0)),
            ([2.0], (0.5, 0.5)),
        ):
            ind = creator.__dict__[ind_name](genes)
            ind.fitness.values = values
            pop.append(ind)

        weights = numpy.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        chosen = tools.sel_moead(pop, 3, weights)
        assert len(chosen) == 3
        assert {id(ind) for ind in chosen} == {id(pop[0]), id(pop[1]), id(pop[2])}
    finally:
        del creator.__dict__[fit_name]
        del creator.__dict__[ind_name]


def test_sel_moead_returns_exact_sel_count_with_crowding_fill():
    fit_name = "MOEAD_FILL_FIT"
    ind_name = "MOEAD_FILL_IND"
    creator.create_type(fit_name, Fitness, weights=(-1.0, -1.0))
    creator.create_type(ind_name, list, fitness=creator.__dict__[fit_name])
    try:
        pop = []
        for genes, values in (
            ([0.0], (0.01, 0.02)),
            ([1.0], (0.0, 1.0)),
            ([2.0], (1.0, 0.0)),
            ([3.0], (0.2, 0.8)),
            ([4.0], (0.3, 0.7)),
            ([5.0], (0.4, 0.6)),
            ([6.0], (0.5, 0.5)),
            ([7.0], (2.0, 2.0)),
            ([8.0], (2.1, 2.1)),
            ([9.0], (2.2, 2.2)),
        ):
            ind = creator.__dict__[ind_name](genes)
            ind.fitness.values = values
            pop.append(ind)

        weights = numpy.array([[1.0, 0.0], [0.0, 1.0]])
        chosen = tools.sel_moead(pop, 6, weights)
        assert len(chosen) == 6
    finally:
        del creator.__dict__[fit_name]
        del creator.__dict__[ind_name]


def test_sel_moead_prefers_lower_front_for_subproblem():
    fit_name = "MOEAD_RANK_FIT"
    ind_name = "MOEAD_RANK_IND"
    creator.create_type(fit_name, Fitness, weights=(-1.0, -1.0))
    creator.create_type(ind_name, list, fitness=creator.__dict__[fit_name])
    try:
        front0 = []
        for genes, values in (
            ([0.0], (0.0, 1.0)),
            ([1.0], (1.0, 0.0)),
            ([2.0], (0.4, 0.4)),
        ):
            ind = creator.__dict__[ind_name](genes)
            ind.fitness.values = values
            front0.append(ind)

        dominated = creator.__dict__[ind_name]([3.0])
        dominated.fitness.values = (0.5, 0.5)
        dominated2 = creator.__dict__[ind_name]([4.0])
        dominated2.fitness.values = (0.6, 0.6)
        pop = front0 + [dominated, dominated2]

        weights = numpy.array([[0.5, 0.5]])
        chosen = tools.sel_moead(pop, 1, weights)
        assert len(chosen) == 1
        assert id(chosen[0]) == id(front0[2])
    finally:
        del creator.__dict__[fit_name]
        del creator.__dict__[ind_name]


def test_oversize_sel_count_returns_full_pool(multi_obj, make):
    weights = tools.uniform_reference_points(2, 4)
    pop = [make(multi_obj, [float(i)], (float(i) * 0.1, 1.0 - float(i) * 0.1)) for i in range(4)]
    chosen = tools.sel_moead(pop, 8, weights)
    assert len(chosen) == 4
    assert {id(ind) for ind in chosen} == {id(ind) for ind in pop}


def test_moead_zdt1_tchebycheff():
    creator.create_type(FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        dimensions = 5
        bound_low, bound_up = 0.0, 1.0
        survivors = 16
        generations = 100
        weights = tools.uniform_reference_points(2, ref_ppo=12)

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
        toolbox.register("select", tools.sel_moead, weights=weights)

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


def test_moead_zdt1_pbi_memory():
    creator.create_type(FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        dimensions = 5
        bound_low, bound_up = 0.0, 1.0
        survivors = 16
        generations = 100
        weights = tools.uniform_reference_points(2, ref_ppo=12)

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
        toolbox.register("select", tools.SelMOEADWithMemory(weights, scalarization="pbi"))

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

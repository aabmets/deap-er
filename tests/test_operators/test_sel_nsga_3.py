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
from deap_er.private.operators.sel_nsga_3_helpers import (
    associate_to_niche,
    find_extreme_points,
    find_intercepts,
)

FIT = "NSGA3_FIT"
IND = "NSGA3_IND"
HV_THRESHOLD = 116.0  # 120.777 is the optimal value


def test_nsga3_with_memory_updates_reference_points(multi_obj, make):
    ref_points = tools.uniform_reference_points(2, 4)
    select = tools.SelNSGA3WithMemory(ref_points)

    tools.rng.seed(7)
    for _ in range(2):
        population = [
            make(multi_obj, [tools.rng.random()], (tools.rng.random(), tools.rng.random()))
            for _ in range(12)
        ]
        select(population, 6)

    assert numpy.all(numpy.isfinite(select.best_point))
    assert numpy.all(numpy.isfinite(select.worst_point))
    assert select.extreme_points is not None


def test_nsga3_associates_points_to_the_nearest_niche():
    fitness = numpy.array(
        [
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.0],
            [0.2, 0.8],
            [0.8, 0.2],
        ],
        dtype=float,
    )
    ref_points = numpy.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    best_point = numpy.zeros(2, dtype=float)
    intercepts = numpy.ones(2, dtype=float)

    niches, distances = associate_to_niche(fitness, ref_points, best_point, intercepts)

    assert niches.tolist() == [2, 1, 0, 2, 0]
    assert distances == pytest.approx([0.0, 0.0, 0.0, 0.2, 0.2], abs=1e-15)


def test_nsga3_constant_objective_uses_absolute_intercepts():
    fit_name = "NSGA3_CONST_FIT"
    ind_name = "NSGA3_CONST_IND"
    creator.create_type(fit_name, Fitness, weights=(-1.0, -1.0))
    creator.create_type(ind_name, list, fitness=creator.__dict__[fit_name])
    try:
        pop = []
        for i in range(5):
            ind = creator.__dict__[ind_name]([float(i), 1.0])
            ind.fitness.values = (float(i), 1.0)
            pop.append(ind)

        fitness = numpy.array([ind.fitness.wvalues for ind in pop], dtype=float) * -1
        best = numpy.min(fitness, axis=0)
        worst = numpy.max(fitness, axis=0)
        ext = find_extreme_points(fitness, best, None)
        intercepts = find_intercepts(ext, best, worst, worst)
        refs = tools.uniform_reference_points(2, 4)
        niches, _ = associate_to_niche(fitness, refs, best, intercepts)

        assert niches.tolist() != [0, 0, 0, 0, 0]
        assert niches[0] != niches[-1]

        best_shift = numpy.array([-10.0, 0.0])
        extreme = numpy.array([[-9.0, 0.0], [-10.0, 10.0]])
        far_worst = numpy.array([100.0, 100.0])
        got = find_intercepts(extreme, best_shift, far_worst, far_worst)
        plane = numpy.linalg.solve(extreme - best_shift, numpy.ones(2))
        assert numpy.allclose(got, 1.0 / plane + best_shift)
    finally:
        del creator.__dict__[fit_name]
        del creator.__dict__[ind_name]


def test_empty_population_returns_empty():
    refs = numpy.array([[1.0, 0.0], [0.0, 1.0]])
    assert tools.sel_nsga_3([], 3, refs) == []


def test_nsga3_oversize_sel_count_returns_unique_pool():
    fit_name = "NSGA3_OVER_FIT"
    ind_name = "NSGA3_OVER_IND"
    creator.create_type(fit_name, Fitness, weights=(-1.0, -1.0))
    creator.create_type(ind_name, list, fitness=creator.__dict__[fit_name])
    try:
        pop = []
        for i, val in enumerate([(0.0, 1.0), (0.3, 0.7), (0.6, 0.4), (1.0, 0.0)]):
            ind = creator.__dict__[ind_name]([float(i)])
            ind.fitness.values = val
            pop.append(ind)

        refs = tools.uniform_reference_points(2, 4)
        chosen = tools.sel_nsga_3(pop, 8, refs)
        assert len(chosen) == 4
        assert {id(ind) for ind in chosen} == {id(ind) for ind in pop}
    finally:
        del creator.__dict__[fit_name]
        del creator.__dict__[ind_name]


def test_nsga3():
    creator.create_type(FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        dimensions = 5
        bound_low, bound_up = 0.0, 1.0
        survivors = 16
        generations = 100

        ref_points = tools.uniform_reference_points(2, ref_ppo=12)

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
        toolbox.register("select", tools.sel_nsga_3, ref_points=ref_points)

        toolbox.register("evaluate", tools.bm_zdt_1)

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
        for ind in pop:
            assert not any(numpy.asarray(ind) < bound_low)
            assert not any(numpy.asarray(ind) > bound_up)
    finally:
        del creator.__dict__[FIT]
        del creator.__dict__[IND]

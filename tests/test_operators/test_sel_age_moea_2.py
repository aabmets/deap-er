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
from deap_er.private.operators.sel_age_moea_2_helpers import (
    estimate_curvature_nr,
    geodesic_distance,
    project_on_manifold,
)

FIT = "AGE2_FIT"
IND = "AGE2_IND"
HV_THRESHOLD = 110.0


def test_empty_population_returns_empty():
    assert tools.sel_age_moea_2([], 3) == []


def test_newton_raphson_sphere():
    point = numpy.array([1.0 / numpy.sqrt(2.0), 1.0 / numpy.sqrt(2.0)])
    p = estimate_curvature_nr(point, 2)
    assert p == pytest.approx(2.0, abs=0.05)


def test_newton_raphson_flat_fallback():
    point = numpy.array([0.0, 0.0])
    p = estimate_curvature_nr(point, 2)
    assert p == 1.0


def test_manifold_projection():
    point = numpy.array([0.5, 0.5])
    projected = project_on_manifold(point, 2.0)
    assert numpy.sum(projected**2) == pytest.approx(1.0, abs=1e-6)


def test_geodesic_symmetry():
    a = numpy.array([0.0, 1.0])
    b = numpy.array([1.0, 0.0])
    d_ab = geodesic_distance(a, b, 2.0)
    d_ba = geodesic_distance(b, a, 2.0)
    assert d_ab == pytest.approx(d_ba)
    assert geodesic_distance(a, a, 2.0) == 0.0


def test_geodesic_flat_case():
    a = numpy.array([0.0, 1.0])
    b = numpy.array([1.0, 0.0])
    d_geo = geodesic_distance(a, b, 1.0)
    d_eucl = numpy.linalg.norm(a - b)
    assert d_geo == pytest.approx(d_eucl)


def test_memory_updates_curvature(multi_obj, make):
    select = tools.SelAGE2WithMemory()
    tools.rng.seed(5)
    for _ in range(2):
        pop = [
            make(multi_obj, [tools.rng.random()], (tools.rng.random(), tools.rng.random()))
            for _ in range(12)
        ]
        select(pop, 6)
    assert numpy.isfinite(select.curvature)
    assert numpy.all(numpy.isfinite(select.best_point))


def test_memory_updates_on_exact_front_fill():
    fit_name = "AGE2_MEM_FIT"
    ind_name = "AGE2_MEM_IND"
    creator.create_type(fit_name, Fitness, weights=(-1.0, -1.0))
    creator.create_type(ind_name, list, fitness=creator.__dict__[fit_name])
    try:
        pop = []
        for genes, values in (([0.0], (0.0, 1.0)), ([1.0], (1.0, 0.0)), ([2.0], (0.5, 0.5))):
            ind = creator.__dict__[ind_name](genes)
            ind.fitness.values = values
            pop.append(ind)

        select = tools.SelAGE2WithMemory()
        assert select.best_point.size == 0
        chosen = select(pop, 3)
        assert len(chosen) == 3
        assert select.best_point.size > 0
        assert numpy.all(numpy.isfinite(select.best_point))
        assert numpy.isfinite(select.curvature)
    finally:
        del creator.__dict__[fit_name]
        del creator.__dict__[ind_name]


def test_best_point_merges_without_worst_point():
    fit_name = "AGE2_BEST_FIT"
    ind_name = "AGE2_BEST_IND"
    creator.create_type(fit_name, Fitness, weights=(-1.0, -1.0))
    creator.create_type(ind_name, list, fitness=creator.__dict__[fit_name])
    try:
        pop = []
        for genes, values in (([0.0], (0.2, 0.8)), ([1.0], (0.8, 0.2))):
            ind = creator.__dict__[ind_name](genes)
            ind.fitness.values = values
            pop.append(ind)

        select = tools.SelAGE2WithMemory()
        select.best_point = numpy.array([[0.0, 0.0]])
        select(pop, 1)
        assert select.best_point.reshape(-1).tolist() == pytest.approx([0.0, 0.0])
        assert select.worst_point.reshape(-1).tolist() == pytest.approx([0.8, 0.8])
    finally:
        del creator.__dict__[fit_name]
        del creator.__dict__[ind_name]


def test_worst_point_ignores_dominated_in_pool():
    fit_name = "AGE2_POOL_FIT"
    ind_name = "AGE2_POOL_IND"
    creator.create_type(fit_name, Fitness, weights=(-1.0, -1.0))
    creator.create_type(ind_name, list, fitness=creator.__dict__[fit_name])
    try:
        parents = []
        for genes, values in (([0.0], (0.0, 1.0)), ([1.0], (1.0, 0.0)), ([2.0], (0.4, 0.6))):
            ind = creator.__dict__[ind_name](genes)
            ind.fitness.values = values
            parents.append(ind)

        offspring = []
        for genes, values in (([10.0], (5.0, 5.0)), ([11.0], (6.0, 6.0))):
            ind = creator.__dict__[ind_name](genes)
            ind.fitness.values = values
            offspring.append(ind)

        pool = parents + offspring
        select = tools.SelAGE2WithMemory()
        select(parents, len(parents))
        select(pool, len(parents))

        assert select.worst_point.reshape(-1).tolist() == pytest.approx([1.0, 1.0])
        assert numpy.all(numpy.isfinite(select.worst_point))
    finally:
        del creator.__dict__[fit_name]
        del creator.__dict__[ind_name]


def test_merge_worst_ignores_non_finite_anchor():
    from deap_er.private.operators.sel_age_moea_2_anchors import merge_worst

    fitness = numpy.array([[1.0, 2.0], [3.0, 4.0]])
    got = merge_worst(fitness, numpy.array([numpy.inf, numpy.inf]))
    assert got.tolist() == pytest.approx([3.0, 4.0])


def test_normalize_front_uses_front_local_worst():
    from deap_er.private.operators.sel_age_moea_2_anchors import normalize_front

    fitness = numpy.array([[0.0, 1.0], [1.0, 0.0]])
    best = numpy.array([0.0, 0.0])
    front_worst = numpy.array([2.0, 2.0])
    _, _, _, tight = normalize_front(
        fitness, best, numpy.array([0.5, 0.5]), None, front_worst=front_worst
    )
    _, _, _, loose = normalize_front(
        fitness, best, numpy.array([5.0, 5.0]), None, front_worst=front_worst
    )
    assert tight.tolist() == pytest.approx([2.0, 2.0])
    assert loose.tolist() == pytest.approx([1.0, 1.0])


def test_age2_oversize_geometry_uses_first_front_only():
    fit_name = "AGE2_OVERSIZE_FIT"
    ind_name = "AGE2_OVERSIZE_IND"
    creator.create_type(fit_name, Fitness, weights=(-1.0, -1.0))
    creator.create_type(ind_name, list, fitness=creator.__dict__[fit_name])
    try:
        front0 = []
        for genes, values in (([0.0], (0.0, 1.0)), ([1.0], (1.0, 0.0))):
            ind = creator.__dict__[ind_name](genes)
            ind.fitness.values = values
            front0.append(ind)

        dominated = creator.__dict__[ind_name]([2.0])
        dominated.fitness.values = (5.0, 5.0)
        pop = front0 + [dominated]

        full = tools.SelAGE2WithMemory()
        full(pop, len(pop))

        reference = tools.SelAGE2WithMemory()
        reference(front0, len(front0))

        assert full.curvature == reference.curvature
        assert full.extreme_points is not None
        assert reference.extreme_points is not None
        assert numpy.allclose(full.extreme_points, reference.extreme_points)
    finally:
        del creator.__dict__[fit_name]
        del creator.__dict__[ind_name]


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

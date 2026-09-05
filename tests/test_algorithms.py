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
from deap_er import Fitness, Toolbox, creator, tools

FITCLSNAME = "FIT_TYPE"
INDCLSNAME = "IND_TYPE"

HV_THRESHOLD = 116.0  # 120.777 is the optimal value


def setup_func_single_obj():
    creator.create_type(FITCLSNAME, Fitness, weights=(-1.0,))
    creator.create_type(INDCLSNAME, list, fitness=creator.__dict__[FITCLSNAME])


def setup_func_multi_obj_numpy():
    creator.create_type(FITCLSNAME, Fitness, weights=(-1.0, -1.0))
    creator.create_type(INDCLSNAME, numpy.ndarray, fitness=creator.__dict__[FITCLSNAME])


def teardown_func():
    del creator.__dict__[FITCLSNAME]
    del creator.__dict__[INDCLSNAME]


def test_standard_cma():
    setup_func_single_obj()

    dimensions = 5
    strategy = tools.Strategy(centroid=[0.0] * dimensions, sigma=1.0)

    toolbox = Toolbox()
    toolbox.register("evaluate", tools.bm_sphere)
    toolbox.register("generate", strategy.generate, creator.__dict__[INDCLSNAME])
    toolbox.register("update", strategy.update)

    pop, _ = tools.ea_generate_update(toolbox, generations=100)
    (best,) = tools.sel_best(pop, sel_count=1)

    assert best.fitness.values < (1e-8,)

    teardown_func()


def test_mo_cma_es():
    setup_func_multi_obj_numpy()

    def distance(feasible_ind, original_ind):
        return sum((f - o) ** 2 for f, o in zip(feasible_ind, original_ind, strict=False))

    def closest_feasible(individual):
        feasible_ind = numpy.array(individual)
        feasible_ind = numpy.maximum(bound_low, feasible_ind)
        feasible_ind = numpy.minimum(bound_up, feasible_ind)
        return feasible_ind

    def valid(individual):
        return not (any(individual < bound_low) or any(individual > bound_up))

    dimensions = 5
    bound_low, bound_up = 0.0, 1.0
    offsprings = 10
    survivors = 10
    generations = 500

    tools.rng.seed(128)

    toolbox = Toolbox()
    toolbox.register("evaluate", tools.bm_zdt_1)
    toolbox.decorate(
        "evaluate", tools.ClosestValidPenalty(valid, closest_feasible, 1.0e6, distance)
    )

    population = [
        creator.__dict__[INDCLSNAME](
            [tools.rng.uniform(bound_low, bound_up) for _ in range(dimensions)]
        )
        for _ in range(survivors)
    ]
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    strategy = tools.StrategyMultiObjective(
        population, sigma=1.0, survivors=survivors, offsprings=offsprings
    )
    toolbox.register("generate", strategy.generate, creator.__dict__[INDCLSNAME])
    toolbox.register("update", strategy.update)

    for _gen in range(generations):
        population = toolbox.generate()

        fitness = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitness, strict=False):
            ind.fitness.values = fit

        toolbox.update(population)

    num_valid = 0
    for ind in strategy.parents:
        dist = distance(closest_feasible(ind), ind)
        if numpy.isclose(dist, 0.0, rtol=1.0e-5, atol=1.0e-5):
            num_valid += 1
    assert num_valid >= len(strategy.parents)

    hv = tools.hypervolume(strategy.parents, [11.0, 11.0])
    assert hv > HV_THRESHOLD
    teardown_func()

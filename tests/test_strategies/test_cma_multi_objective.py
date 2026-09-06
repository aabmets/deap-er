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

MO_FIT = "MOCMA_FIT"
MO_IND = "MOCMA_IND"


@pytest.fixture
def population():
    creator.create_type(MO_FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(MO_IND, numpy.ndarray, fitness=creator.__dict__[MO_FIT])

    tools.rng.seed(3)
    choices = numpy.array([[tools.rng.uniform(0.0, 1.0) for _ in range(4)] for _ in range(6)])
    individuals = [creator.__dict__[MO_IND](x) for x in choices]
    for ind in individuals:
        ind.fitness.values = tools.bm_zdt_1(ind)

    yield individuals

    del creator.__dict__[MO_FIT]
    del creator.__dict__[MO_IND]


def test_defaults_derive_from_the_population(population):
    strategy = tools.StrategyMultiObjective(population, sigma=1.0)

    assert strategy.mu == len(population)
    assert strategy.lamb == 1
    assert strategy.thresh_sr == 0.44
    assert strategy.psucc == [strategy.tgt_sr] * len(population)


def test_kwargs_override_the_defaults(population):
    strategy = tools.StrategyMultiObjective(
        population, sigma=1.0, survivors=4, offsprings=8, thresh_sr=0.5
    )

    assert strategy.mu == 4
    assert strategy.lamb == 8
    assert strategy.thresh_sr == 0.5


def test_compute_params_can_be_called_again(population):
    strategy = tools.StrategyMultiObjective(population, sigma=1.0, offsprings=2)

    strategy.compute_params(offsprings=5, survivors=3)

    assert strategy.lamb == 5
    assert strategy.mu == 3


def test_stall_alpha_includes_covariance_learning_rate(population):
    strategy = tools.StrategyMultiObjective(population, sigma=1.0)
    cc, c_cov = strategy.th_cum, strategy.cm_learn_rate
    published = (1 - c_cov) + c_cov * cc * (2 - cc)
    dim = len(population[0])
    child = creator.__dict__[MO_IND](numpy.array(population[0], copy=True))
    child.fitness.values = population[0].fitness.values
    child.ps_ = ("o", 0)
    identity = numpy.eye(dim)
    path = numpy.ones(dim)
    big = [identity.copy()]
    inv = [identity.copy()]
    strategy.update_chosen_offspring([child], [1.0], [1.0], inv, big, [path.copy()], [1.0])
    decayed = (1.0 - cc) * path
    _, expected = strategy.rank_one_update(
        identity.copy(), identity.copy(), published, c_cov, decayed
    )
    assert big[0] == pytest.approx(expected)


def test_rank_one_update_runs_for_all_negative_path(population):
    strategy = tools.StrategyMultiObjective(population, sigma=1.0)
    identity = numpy.eye(2)
    _, negative = strategy.rank_one_update(
        identity.copy(), identity.copy(), 0.9, 0.05, numpy.array([-1.0, -1.0])
    )
    _, positive = strategy.rank_one_update(
        identity.copy(), identity.copy(), 0.9, 0.05, numpy.array([1.0, 1.0])
    )
    assert not numpy.allclose(negative, identity)
    assert negative == pytest.approx(positive)


def test_generate_tags_offspring_with_their_parent(population):
    strategy = tools.StrategyMultiObjective(population, sigma=1.0, survivors=6, offsprings=6)

    offspring = strategy.generate(creator.__dict__[MO_IND])

    assert len(offspring) == 6
    assert all(ind.ps_[0] == "o" for ind in offspring)
    assert all(0 <= ind.ps_[1] < len(population) for ind in offspring)


def test_generate_respects_box_bounds():
    creator.create_type(MO_FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(MO_IND, numpy.ndarray, fitness=creator.__dict__[MO_FIT])
    try:
        tools.rng.seed(3)
        parents = [creator.__dict__[MO_IND]([0.2, 0.3, 0.4]) for _ in range(4)]
        for parent in parents:
            parent.fitness.values = tools.bm_zdt_1(parent)
        strategy = tools.StrategyMultiObjective(
            parents, sigma=8.0, survivors=4, offsprings=4, low=0.0, up=1.0
        )
        children = strategy.generate(creator.__dict__[MO_IND])
        assert len(children) == 4
        for child in children:
            genes = numpy.asarray(child)
            assert numpy.all(genes >= 0.0)
            assert numpy.all(genes <= 1.0)
    finally:
        del creator.__dict__[MO_FIT]
        del creator.__dict__[MO_IND]


def test_mo_cma_es():
    smoke_fit = "MOCMA_SMOKE_FIT"
    smoke_ind = "MOCMA_SMOKE_IND"
    creator.create_type(smoke_fit, Fitness, weights=(-1.0, -1.0))
    creator.create_type(smoke_ind, numpy.ndarray, fitness=creator.__dict__[smoke_fit])
    try:

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
            creator.__dict__[smoke_ind](
                [tools.rng.uniform(bound_low, bound_up) for _ in range(dimensions)]
            )
            for _ in range(survivors)
        ]
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)

        strategy = tools.StrategyMultiObjective(
            population, sigma=1.0, survivors=survivors, offsprings=offsprings
        )
        toolbox.register("generate", strategy.generate, creator.__dict__[smoke_ind])
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
        assert hv > 116.0
    finally:
        del creator.__dict__[smoke_fit]
        del creator.__dict__[smoke_ind]


def test_multi_child_parent_sigma_updates_once(population):
    parents = population[:2]
    for parent, val in zip(parents, [(0.0, 1.0), (1.0, 0.0)], strict=True):
        parent.fitness.values = val
    mo = tools.StrategyMultiObjective(parents, sigma=1.0, survivors=2, offsprings=3)
    for i, parent in enumerate(mo.parents):
        parent.ps_ = "p", i
    cls = type(parents[0])
    kids = []
    for p_idx in (0, 0, 1):
        child = cls(numpy.array(parents[p_idx], copy=True))
        child.fitness.values = (3.0, 3.0)
        child.ps_ = "o", p_idx
        kids.append(child)
    mo.update(kids)
    assert mo.psucc[0] == pytest.approx(mo.psucc[1])
    assert mo.sigmas[0] == pytest.approx(mo.sigmas[1])


def test_rank_one_update_scales_when_path_is_zero(population):
    identity = numpy.identity(2)
    inv, big_a = tools.StrategyMultiObjective.rank_one_update(
        identity.copy(), identity.copy(), 0.9, 0.1, numpy.zeros(2)
    )
    scale = numpy.sqrt(0.9)
    assert big_a == pytest.approx(scale * identity)
    assert inv == pytest.approx(identity / scale)


def test_invalid_parent_fitness_promotes_evaluated_offspring():
    fit, name = "MOCMA_UNEV_FIT", "MOCMA_UNEV_IND"
    creator.create_type(fit, Fitness, weights=(-1.0, -1.0))
    creator.create_type(name, list, fitness=creator.__dict__[fit])
    try:
        uneval = [creator.__dict__[name]([0.0, 0.0]), creator.__dict__[name]([1.0, 1.0])]
        mo = tools.StrategyMultiObjective(uneval, sigma=1.0)
        tb = Toolbox()
        tb.register("generate", mo.generate, creator.__dict__[name])
        tb.register("update", mo.update)
        tb.register("evaluate", lambda ind: (float(ind[0]), float(ind[1])))
        tools.ea_generate_update(tb, generations=1)
        assert mo.parents
        assert all(ind.fitness.is_valid() for ind in mo.parents)
        assert all(ind.ps_[0] == "o" for ind in mo.parents)
    finally:
        del creator.__dict__[fit]
        del creator.__dict__[name]

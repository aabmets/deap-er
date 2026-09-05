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
from deap_er import base, creator, tools

MO_FIT = "MOCMA_FIT"
MO_IND = "MOCMA_IND"


@pytest.fixture
def population():
    creator.create(MO_FIT, base.Fitness, weights=(-1.0, -1.0))
    creator.create(MO_IND, numpy.ndarray, fitness=creator.__dict__[MO_FIT])

    tools.seed(3)
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

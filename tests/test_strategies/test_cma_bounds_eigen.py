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
from deap_er import Fitness, creator, tools
from deap_er.private.strategies.clip_samples import RAW_SAMPLE_ATTR, raw_sample

FIT = "CMA_BE_FIT"
IND = "CMA_BE_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(FIT, Fitness, weights=(1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    yield creator.__dict__[IND]
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def _assert_finite_state(strategy):
    assert numpy.all(numpy.isfinite(strategy.big_c))
    assert numpy.all(numpy.isfinite(strategy.diag_d))
    assert numpy.all(strategy.diag_d > 0.0)
    assert numpy.isfinite(strategy.cond)
    assert strategy.cond > 0.0
    assert numpy.isfinite(strategy.sigma)


def test_singular_cm_init_gives_finite_positive_eigen_state(ind_cls):
    v = numpy.array([1.0, 1.0, 1e-9])
    strategy = tools.Strategy([0.0, 0.0, 0.0], 1.0, cm_init=numpy.outer(v, v))
    _assert_finite_state(strategy)
    tools.rng.seed(3)
    population = strategy.generate(ind_cls)
    for ind in population:
        assert numpy.all(numpy.isfinite(numpy.asarray(ind, dtype=float)))
        ind.fitness.values = (-sum(x * x for x in ind),)
    strategy.update(population)
    _assert_finite_state(strategy)


def test_clip_keeps_unclipped_draw_and_evaluates_clipped_point(ind_cls):
    strategy = tools.Strategy([1.0, 1.0], 1.0, low=0.0, up=1.0)
    tools.rng.seed(5)
    population = strategy.generate(ind_cls)
    clipped = [ind for ind in population if hasattr(ind, RAW_SAMPLE_ATTR)]
    assert clipped
    for ind in population:
        assert all(0.0 <= x <= 1.0 for x in ind)
    for ind in clipped:
        genes = numpy.asarray(ind, dtype=float)
        raw = raw_sample(ind)
        assert numpy.array_equal(numpy.clip(raw, 0.0, 1.0), genes)
        assert not numpy.array_equal(raw, genes)


@pytest.mark.parametrize("seed", [0, 1])
def test_clip_corner_optimum_keeps_covariance_bounded(ind_cls, seed):
    tools.rng.seed(seed)
    strategy = tools.Strategy([0.5] * 11, 0.3, offsprings=16, low=0.0, up=1.0)
    for _ in range(300):
        population = strategy.generate(ind_cls)
        for ind in population:
            ind.fitness.values = (sum(ind),)
        strategy.update(population)
        _assert_finite_state(strategy)
        assert strategy.cond < 1e6
    assert numpy.allclose(strategy.centroid, 1.0)


def test_clip_corner_optimum_separable_keeps_variance_bounded(ind_cls):
    tools.rng.seed(0)
    strategy = tools.StrategySeparable([0.5] * 11, 0.3, offsprings=16, low=0.0, up=1.0)
    for _ in range(300):
        population = strategy.generate(ind_cls)
        for ind in population:
            ind.fitness.values = (sum(ind),)
        strategy.update(population)
        assert numpy.all(numpy.isfinite(strategy.big_c))
        assert strategy.cond < 1e6
    assert numpy.allclose(strategy.centroid, 1.0)

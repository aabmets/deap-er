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
import pytest
from deap_er import Fitness, creator, tools
from deap_er.private.strategies.restart_ops import resize_offsprings

FIT = "CMA_PIN_FIT"
IND = "CMA_PIN_IND"

RATES = {"rank_one": 0.05, "rank_mu": 0.1, "ss_cum": 0.4, "ss_dmp": 3.0, "cm_cum": 0.3}


@pytest.fixture
def ind_cls():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    yield creator.__dict__[IND]
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def _rates(strategy):
    return {name: getattr(strategy, name) for name in RATES}


@pytest.mark.parametrize("cls", [tools.Strategy, tools.StrategySeparable])
def test_reset_state_keeps_constructor_hyperparams(cls):
    strategy = cls([0.0] * 4, 1.0, offsprings=6, survivors=2, weights="equal", **RATES)
    strategy.reset_state([1.0] * 4, 0.5)
    assert (strategy.lamb, strategy.mu) == (6, 2)
    assert list(strategy.weights) == [0.5, 0.5]
    assert _rates(strategy) == RATES


@pytest.mark.parametrize("cls", [tools.Strategy, tools.StrategySeparable])
def test_reset_state_rederives_only_unpinned_values(cls):
    strategy = cls([0.0] * 4, 1.0, offsprings=6, rank_mu=0.1, weights="linear")
    strategy.reset_state([0.0] * 4, 1.0, offsprings=10, ss_dmp=2.0)
    assert (strategy.lamb, strategy.mu) == (10, 5)
    assert strategy.rank_mu == pytest.approx(0.1)
    assert strategy.ss_dmp == pytest.approx(2.0)
    assert strategy.weights[0] == pytest.approx(4.5 / 12.5)


def test_pinned_survivors_are_capped_by_a_smaller_lambda_and_restored_later():
    strategy = tools.Strategy([0.0] * 4, 1.0, offsprings=10, survivors=5)
    strategy.reset_state([0.0] * 4, 1.0, offsprings=4)
    assert strategy.mu == 4
    strategy.reset_state([0.0] * 4, 1.0, offsprings=12)
    assert strategy.mu == 5


def test_explicit_survivors_above_offsprings_still_raise():
    strategy = tools.Strategy([0.0] * 4, 1.0)
    with pytest.raises(ValueError, match="survivors"):
        strategy.compute_params(offsprings=4, survivors=5)


def test_resize_offsprings_keeps_rates_and_does_not_pin_the_capped_mu():
    strategy = tools.Strategy([0.0] * 4, 1.0, offsprings=8, **RATES)
    resize_offsprings(strategy, 3)
    assert (strategy.lamb, strategy.mu) == (3, 3)
    assert _rates(strategy) == RATES
    strategy.reset_state([0.0] * 4, 1.0, offsprings=8)
    assert strategy.mu == 4


def test_restart_keeps_user_hyperparams(ind_cls):
    strategy = tools.Strategy([0.0] * 3, 1.0, offsprings=6, weights="equal", **RATES)
    restart = tools.RestartStrategy(strategy, mode="ipop", budget=1000)
    population = restart.generate(ind_cls)
    for individual in population:
        individual.fitness.values = tools.bm_sphere(individual)
    restart.update(population)
    restart.restart()
    assert strategy.lamb == 12
    assert strategy.mu == 6
    assert list(strategy.weights) == pytest.approx([1 / 6] * 6)
    assert _rates(strategy) == RATES


def test_one_plus_lambda_reset_state_keeps_hyperparams(ind_cls):
    parent = ind_cls([0.0, 0.0])
    parent.fitness.values = (0.0,)
    pins = {"offsprings": 3, "ss_dmp": 2.5, "tgt_sr": 0.3, "cm_learn_rate": 0.2, "thresh_sr": 0.5}
    strategy = tools.StrategyOnePlusLambda(parent, 1.0, **pins)
    strategy.reset_state(ind_cls([1.0, 1.0]), 0.5, offsprings=5)
    assert strategy.lamb == 5
    assert strategy.ss_dmp == 2.5
    assert strategy.tgt_sr == 0.3
    assert strategy.cm_learn_rate == 0.2
    assert strategy.thresh_sr == 0.5


def test_multi_objective_reset_state_keeps_hyperparams():
    creator.create_type(FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        parents = [creator.__dict__[IND]([float(i), 0.0]) for i in range(3)]
        for parent in parents:
            parent.fitness.values = (float(len(parents)), 0.0)
        pins = {"ss_dmp": 2.5, "tgt_sr": 0.3, "th_cum": 0.4, "cm_learn_rate": 0.2}
        strategy = tools.StrategyMultiObjective(parents, 1.0, offsprings=3, **pins)
        strategy.reset_state(parents, 0.5, offsprings=2, survivors=2)
        assert (strategy.lamb, strategy.mu) == (2, 2)
        assert (strategy.ss_dmp, strategy.tgt_sr, strategy.th_cum) == (2.5, 0.3, 0.4)
        assert strategy.cm_learn_rate == 0.2
    finally:
        del creator.__dict__[FIT]
        del creator.__dict__[IND]

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
from typing import Any

import pytest
from deap_er import Fitness, creator, tools

FIT = "OPL_FIT"
IND = "OPL_IND"


def _setup():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    return creator.__dict__[IND]


def _teardown():
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def test_one_plus_lambda_requires_fitness():
    parent: Any = [0.0, 0.0]
    with pytest.raises(TypeError, match="fitness"):
        tools.StrategyOnePlusLambda(parent, 1.0)


def test_one_plus_lambda_reset_state_and_high_success_update():
    ind_cls = _setup()
    try:
        parent = ind_cls([1.0, 1.0])
        parent.fitness.values = (2.0,)
        strategy = tools.StrategyOnePlusLambda(parent, sigma=0.5, offsprings=2, thresh_sr=0.0)
        strategy.reset_state(parent, 0.2, offsprings=3)
        assert strategy.sigma == 0.2
        assert strategy.lamb == 3

        better = ind_cls([0.0, 0.0])
        better.fitness.values = (0.1,)
        worse = ind_cls([2.0, 2.0])
        worse.fitness.values = (4.0,)
        strategy.update([better, worse])
        assert list(strategy.parent) == [0.0, 0.0]
        assert strategy.sigma > 0.0
    finally:
        _teardown()


def test_update_invalid_parent_adopts_best_without_fake_success():
    ind_cls = _setup()
    try:
        parent = ind_cls([0.0, 0.0])
        del parent.fitness.values
        strategy = tools.StrategyOnePlusLambda(parent, sigma=1.0, offsprings=3)
        sigma0 = strategy.sigma
        psucc0 = strategy.psucc
        cov0 = strategy.big_c.copy()
        best = ind_cls([1.0, 0.0])
        best.fitness.values = (5.0,)
        mid = ind_cls([2.0, 0.0])
        mid.fitness.values = (10.0,)
        worst = ind_cls([3.0, 0.0])
        worst.fitness.values = (20.0,)
        strategy.update([worst, best, mid])
        assert list(strategy.parent) == [1.0, 0.0]
        assert strategy.parent.fitness.values == (5.0,)
        assert strategy.psucc == psucc0
        assert strategy.sigma == sigma0
        assert strategy.big_c == pytest.approx(cov0)
    finally:
        _teardown()

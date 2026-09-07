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
from deap_er import Fitness, creator, tools
from deap_er.private.strategies.restart_ops import (
    apply_strategy_restart,
    resize_offsprings,
    set_strategy_sigma,
)

FIT = "ROP_FIT"
IND = "ROP_IND"
MO_FIT = "ROP_MO_FIT"
MO_IND = "ROP_MO_IND"


def _so_types():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    return creator.__dict__[IND]


def _mo_types():
    creator.create_type(MO_FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(MO_IND, list, fitness=creator.__dict__[MO_FIT])
    return creator.__dict__[MO_IND]


def _teardown_so():
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def _teardown_mo():
    del creator.__dict__[MO_FIT]
    del creator.__dict__[MO_IND]


def test_set_sigma_and_resize_multi_objective():
    ind_cls = _mo_types()
    try:
        parents = [ind_cls([0.0, 0.0]), ind_cls([1.0, 1.0])]
        for parent in parents:
            parent.fitness.values = (1.0, 1.0)
        strategy = tools.StrategyMultiObjective(parents, sigma=1.0, survivors=2, offsprings=4)

        set_strategy_sigma(strategy, 0.25)
        resize_offsprings(strategy, 1)

        assert strategy.sigmas == [0.25, 0.25]
        assert strategy.lamb == 1
        assert strategy.mu == 1
    finally:
        _teardown_mo()


def test_apply_restart_one_plus_lambda():
    ind_cls = _so_types()
    try:
        parent = ind_cls([0.0, 0.0])
        parent.fitness.values = (1.0,)
        strategy = tools.StrategyOnePlusLambda(parent, sigma=1.0, offsprings=3)
        apply_strategy_restart(
            strategy,
            ind_cls,
            dim=2,
            lamb=5,
            sigma=0.4,
            restart_centroid="initial",
            initial_center=numpy.array([2.0, 3.0]),
            best=None,
        )
        assert strategy.sigma == 0.4
        assert strategy.lamb == 5
        assert list(strategy.parent) == [2.0, 3.0]
        assert not strategy.parent.fitness.is_valid()
    finally:
        _teardown_so()


def test_apply_restart_multi_objective():
    ind_cls = _mo_types()
    try:
        parents = [ind_cls([0.0, 0.0]), ind_cls([1.0, 1.0])]
        for parent in parents:
            parent.fitness.values = (1.0, 2.0)
        strategy = tools.StrategyMultiObjective(parents, sigma=1.0, survivors=2, offsprings=2)
        apply_strategy_restart(
            strategy,
            ind_cls,
            dim=2,
            lamb=1,
            sigma=0.8,
            restart_centroid="random",
            initial_center=numpy.array([0.0, 0.0]),
            best=None,
        )
        assert strategy.sigmas == [0.8]
        assert strategy.lamb == 1
        assert len(strategy.parents) == 1
        assert not strategy.parents[0].fitness.is_valid()
    finally:
        _teardown_mo()

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
from deap_er.private.strategies.restart_common import (
    RunTracker,
    sample_centroid,
    strategy_center,
    strategy_diagnostics,
    strategy_sigma,
)

FIT = "RCM_FIT"
IND = "RCM_IND"


def _types():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    return creator.__dict__[IND]


def _teardown():
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def test_sample_centroid_modes():
    ind_cls = _types()
    try:
        best = ind_cls([9.0, 8.0])
        initial = numpy.array([1.0, 2.0])

        assert numpy.allclose(sample_centroid(2, None, None, "initial", initial, None), initial)
        assert numpy.allclose(sample_centroid(2, None, None, "best", initial, best), [9.0, 8.0])
        custom = sample_centroid(2, None, None, lambda dim: numpy.array([3.0] * dim), initial, None)
        assert numpy.allclose(custom, [3.0, 3.0])

        tools.rng.seed(0)
        unbounded = sample_centroid(2, None, None, "random", initial, None)
        tools.rng.seed(0)
        bounded = sample_centroid(2, 0.0, 1.0, "random", initial, None)
        assert unbounded.shape == (2,)
        assert bounded.shape == (2,)
        assert all(0.0 <= value <= 1.0 for value in bounded)
    finally:
        _teardown()


def test_run_tracker_max_iter_and_condition():
    ind_cls = _types()
    try:
        individual = ind_cls([0.0])
        individual.fitness.values = (1.0,)
        tracker = RunTracker(1, 4, 1.0, tol_fun=0.0)
        tracker.begin_run(4, 1.0, max_iter=2)
        tracker.observe([individual])
        assert tracker.terminate is False
        tracker.observe([individual])
        assert tracker.terminate is True

        tracker.begin_run(4, 1.0, max_iter=None)
        tracker.observe([individual], condition=1e20)
        assert tracker.terminate is True

        tracker.begin_run(4, 1.0)
        tracker.observe([individual], sigma=1e30, largest_eig=1.0)
        assert tracker.terminate is True
    finally:
        _teardown()


def test_run_tracker_stagnation_with_disabled_tol_fun():
    ind_cls = _types()
    try:
        tracker = RunTracker(2, 20, 1.0, stagnation_window=5, tol_fun=0.0)
        tracker.begin_run(20, 1.0)
        for _ in range(250):
            individual = ind_cls([1.0, 1.0])
            individual.fitness.values = (1.0,)
            tracker.observe([individual])
            if tracker.terminate:
                break
        assert tracker.terminate
    finally:
        _teardown()


def test_strategy_center_sigma_and_diagnostics():
    so_cls = _types()
    try:
        parent = so_cls([0.5, 0.5])
        parent.fitness.values = (1.0,)
        standard = tools.Strategy(centroid=[1.0, 2.0], sigma=0.3, offsprings=6)
        one_plus = tools.StrategyOnePlusLambda(parent, sigma=0.7, offsprings=2)

        creator.create_type("RCM_MO_FIT", Fitness, weights=(-1.0, -1.0))
        creator.create_type("RCM_MO_IND", list, fitness=creator.__dict__["RCM_MO_FIT"])
        mo_cls = creator.__dict__["RCM_MO_IND"]
        parents = [mo_cls([0.0, 1.0]), mo_cls([1.0, 0.0])]
        for item in parents:
            item.fitness.values = (1.0, 1.0)
        multi = tools.StrategyMultiObjective(parents, sigma=0.4)

        assert numpy.allclose(strategy_center(standard), [1.0, 2.0])
        assert numpy.allclose(strategy_center(one_plus), [0.5, 0.5])
        assert numpy.allclose(strategy_center(multi), [0.0, 1.0])
        assert strategy_sigma(standard) == 0.3
        assert strategy_sigma(one_plus) == 0.7
        assert strategy_sigma(multi) == 0.4
        assert strategy_diagnostics(standard)[0] is not None
        assert strategy_diagnostics(one_plus) == (None, 0.7, None)
        assert strategy_diagnostics(multi)[1] == 0.4
    finally:
        del creator.__dict__["RCM_MO_FIT"]
        del creator.__dict__["RCM_MO_IND"]
        _teardown()

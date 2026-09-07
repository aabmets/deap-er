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
from deap_er.private.strategies.restart_common import RunTracker

FIT = "RST_FIT"
IND = "RST_IND"


def _setup_min(dim: int = 5, sigma: float = 1.0, offsprings: int = 10):
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    strategy = tools.Strategy(
        centroid=[0.0] * dim,
        sigma=sigma,
        offsprings=offsprings,
        low=-5.0,
        up=5.0,
    )
    toolbox = Toolbox()
    toolbox.register("evaluate", tools.bm_sphere)
    toolbox.register("generate", strategy.generate, creator.__dict__[IND])
    toolbox.register("update", strategy.update)
    return strategy, toolbox


def _teardown():
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def test_strategy_reset_state():
    strategy, toolbox = _setup_min()
    try:
        pop = toolbox.generate()
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)
        toolbox.update(pop)
        assert strategy.update_count == 1
        strategy.reset_state([1.0] * 5, 0.5, offsprings=20)
        assert strategy.update_count == 0
        assert strategy.sigma == 0.5
        assert strategy.lamb == 20
        assert numpy.allclose(strategy.big_c, numpy.identity(5))
        assert numpy.allclose(strategy.pc, 0.0)
    finally:
        _teardown()


def test_run_tracker_stagnation():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        tracker = RunTracker(5, 10, 2.0, stagnation_window=5)
        tracker.begin_run(10, 2.0, max_iter=10_000)
        flat = 1.0
        for _ in range(300):
            ind = creator.__dict__[IND]([flat] * 5)
            ind.fitness.values = (flat,)
            tracker.observe([ind])
            if tracker.terminate:
                break
        assert tracker.terminate
    finally:
        _teardown()


def test_ipop_doubles_lambda_on_restart():
    strategy, toolbox = _setup_min(offsprings=8)
    try:
        restart = tools.RestartStrategy(strategy, mode="ipop", budget=1_000_000)
        toolbox.register("generate", restart.generate, creator.__dict__[IND])
        toolbox.register("update", restart.update)
        restart.generate(creator.__dict__[IND])
        restart._tracker.terminate = True
        assert restart.should_restart()
        restart.restart()
        assert strategy.lamb == 16
        assert strategy.update_count == 0
        assert restart.restart_count == 1
    finally:
        _teardown()


def test_restart_centroid_in_box():
    strategy, _ = _setup_min()
    try:
        restart = tools.RestartStrategy(strategy, mode="ipop", budget=100_000)
        restart._ind_init = creator.__dict__[IND]
        restart._tracker.terminate = True
        restart.restart()
        assert numpy.all(strategy.centroid >= -5.0)
        assert numpy.all(strategy.centroid <= 5.0)
    finally:
        _teardown()


def test_bipop_small_regime_samples_lambda():
    strategy, _ = _setup_min(offsprings=8)
    try:
        restart = tools.RestartStrategy(strategy, mode="bipop", budget=1_000_000)
        restart._lambda_large = 32
        restart._budget_large = 100
        restart._budget_small = 0
        restart._restart_count = 2
        restart._ind_init = creator.__dict__[IND]
        restart._tracker.terminate = True
        restart.restart()
        assert restart.regime == "small"
        assert 8 <= strategy.lamb <= 16
    finally:
        _teardown()


def test_one_plus_lambda_smoke():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        parent = creator.__dict__[IND]([0.0] * 5)
        del parent.fitness.values
        strategy = tools.StrategyOnePlusLambda(parent, sigma=1.0, offsprings=4)
        restart = tools.RestartStrategy(strategy, mode="ipop", budget=2_000)
        toolbox = Toolbox()
        toolbox.register("evaluate", tools.bm_sphere)
        toolbox.register("generate", restart.generate, creator.__dict__[IND])
        toolbox.register("update", restart.update)
        pop, _ = tools.ea_generate_update_restarts(toolbox, restart, log_restarts=False)
        assert restart.evals_used > 0
        assert len(pop) == strategy.lamb
    finally:
        _teardown()


def test_restart_centroid_one_sided_bounds_stay_finite():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        low_only = tools.Strategy([0.5] * 5, sigma=1.0, offsprings=8, low=0.0)
        restart = tools.RestartStrategy(low_only, mode="ipop", budget=100_000)
        restart._ind_init = creator.__dict__[IND]
        restart._tracker.terminate = True
        restart.restart()
        assert numpy.isfinite(low_only.centroid).all()
        assert numpy.all(low_only.centroid >= 0.0)

        up_only = tools.Strategy([0.5] * 5, sigma=1.0, offsprings=8, up=1.0)
        restart = tools.RestartStrategy(up_only, mode="ipop", budget=100_000)
        restart._ind_init = creator.__dict__[IND]
        restart._tracker.terminate = True
        restart.restart()
        assert numpy.isfinite(up_only.centroid).all()
        assert numpy.all(up_only.centroid <= 1.0)
    finally:
        _teardown()


def test_sample_centroid_expands_when_default_box_misses_the_bound():
    from deap_er.private.strategies.restart_common import sample_centroid

    tools.rng.seed(0)
    high_low = sample_centroid(4, 10.0, None, "random", numpy.zeros(4), None)
    assert numpy.isfinite(high_low).all()
    assert numpy.all(high_low >= 10.0)

    low_up = sample_centroid(4, None, -10.0, "random", numpy.zeros(4), None)
    assert numpy.isfinite(low_up).all()
    assert numpy.all(low_up <= -10.0)


def test_budget_cap():
    strategy, toolbox = _setup_min(offsprings=6)
    try:
        restart = tools.RestartStrategy(strategy, mode="ipop", budget=60)
        toolbox.register("generate", restart.generate, creator.__dict__[IND])
        toolbox.register("update", restart.update)
        tools.ea_generate_update_restarts(toolbox, restart, log_restarts=False)
        assert restart.evals_used <= 60
    finally:
        _teardown()

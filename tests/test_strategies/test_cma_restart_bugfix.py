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
import math

import pytest
from deap_er import Fitness, Toolbox, creator, tools
from deap_er.private.strategies.restart_common import RunTracker, scalar_fitness

FIT = "RST_BF_FIT"
IND = "RST_BF_IND"


def _teardown():
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def test_budget_hard_cap_when_not_multiple_of_lambda():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        strategy = tools.Strategy([0.0] * 5, sigma=1.0, offsprings=6)
        restart = tools.RestartStrategy(strategy, mode="ipop", budget=65)
        toolbox = Toolbox()
        toolbox.register("evaluate", tools.bm_sphere)
        toolbox.register("generate", restart.generate, creator.__dict__[IND])
        toolbox.register("update", restart.update)
        tools.ea_generate_update_restarts(toolbox, restart, log_restarts=False)
        assert restart.evals_used == 65
    finally:
        _teardown()


def test_scalar_fitness_uses_weighted_values_for_maximization():
    creator.create_type(FIT, Fitness, weights=(1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        better = creator.__dict__[IND]([0.0])
        better.fitness.values = (10.0,)
        worse = creator.__dict__[IND]([1.0])
        worse.fitness.values = (5.0,)
        assert scalar_fitness(better) == 10.0
        assert scalar_fitness(worse) == 5.0
    finally:
        _teardown()


def test_restart_update_prefers_weighted_best_for_maximization():
    creator.create_type(FIT, Fitness, weights=(1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        strategy = tools.Strategy([0.0] * 3, sigma=1.0, offsprings=2)
        restart = tools.RestartStrategy(strategy, mode="ipop", budget=100)
        better = creator.__dict__[IND]([1.0, 1.0, 1.0])
        better.fitness.values = (10.0,)
        worse = creator.__dict__[IND]([0.0, 0.0, 0.0])
        worse.fitness.values = (5.0,)
        restart.update([worse, better])
        assert restart._best is better
    finally:
        _teardown()


def test_mo_reset_trims_parent_arrays_to_survivors():
    creator.create_type(FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        parents = [creator.__dict__[IND]([0.0, 0.0]) for _ in range(4)]
        for parent in parents:
            del parent.fitness.values
        mo = tools.StrategyMultiObjective(parents, sigma=1.0, offsprings=8, survivors=4)
        mo.reset_state(parents, sigma=0.5, offsprings=2, survivors=2)
        assert mo.mu == 2
        assert len(mo.parents) == 2
        assert len(mo.sigmas) == 2
        assert len(mo.big_a) == 2
    finally:
        _teardown()


def test_first_run_sigma_matches_sigma_large():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        strategy = tools.Strategy([0.0] * 5, sigma=5.0, offsprings=6)
        restart = tools.RestartStrategy(strategy, mode="ipop", budget=100, sigma_large=2.0)
        assert strategy.sigma == 2.0
        assert restart._tracker.sigma0 == 2.0
    finally:
        _teardown()


def test_run_tracker_sigma0_matches_strategy_sigma():
    tracker = RunTracker(5, 10, 3.5)
    tracker.begin_run(10, 3.5)
    assert tracker.sigma0 == 3.5


def test_scalar_fitness_mo_without_key_raises():
    creator.create_type(FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        ind = creator.__dict__[IND]([0.0, 0.0])
        ind.fitness.values = (1.0, 2.0)
        with pytest.raises(ValueError, match="stagnation_key"):
            scalar_fitness(ind)
    finally:
        _teardown()


def test_mo_restart_requires_explicit_stagnation_key():
    creator.create_type(FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        parents = [creator.__dict__[IND]([0.0, 0.0]) for _ in range(4)]
        for parent in parents:
            del parent.fitness.values
        mo = tools.StrategyMultiObjective(parents, sigma=1.0, offsprings=8, survivors=4)
        restart = tools.RestartStrategy(mo, mode="ipop", budget=100)
        ind = creator.__dict__[IND]([0.0, 0.0])
        ind.fitness.values = (1.0, 2.0)
        with pytest.raises(ValueError, match="stagnation_key"):
            restart.update([ind])
    finally:
        _teardown()


def test_mo_restart_accepts_explicit_stagnation_key():
    creator.create_type(FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        parents = [creator.__dict__[IND]([0.1, 0.2]) for _ in range(4)]
        for parent in parents:
            parent.fitness.values = (parent[0] ** 2, parent[1] ** 2)
        mo = tools.StrategyMultiObjective(parents, sigma=0.1, offsprings=4, survivors=4)
        ref = [10.0, 10.0]
        restart = tools.RestartStrategy(
            mo,
            mode="ipop",
            budget=100,
            stagnation_key=lambda ind: tools.hypervolume([ind], ref),
        )

        def evaluate(ind):
            return (ind[0] ** 2, ind[1] ** 2)

        offspring = restart.generate(creator.__dict__[IND])
        for ind in offspring:
            ind.fitness.values = evaluate(ind)
        restart.update(offspring)
        assert restart._best is not None
    finally:
        _teardown()


def test_stagnation_key_must_be_higher_is_better():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        strategy = tools.Strategy([0.0] * 3, sigma=1.0, offsprings=2)
        restart = tools.RestartStrategy(
            strategy,
            mode="ipop",
            budget=100,
            stagnation_key=lambda ind: float(ind.fitness.values[0]),
        )
        better = creator.__dict__[IND]([1.0, 1.0, 1.0])
        better.fitness.values = (10.0,)
        worse = creator.__dict__[IND]([0.0, 0.0, 0.0])
        worse.fitness.values = (5.0,)
        restart.update([worse, better])
        assert restart._best is better
    finally:
        _teardown()


def test_best_fitness_nan_before_first_update():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        strategy = tools.Strategy([0.0] * 5, sigma=1.0, offsprings=6)
        restart = tools.RestartStrategy(strategy, mode="ipop", budget=100)
        assert math.isnan(restart.best_fitness)
    finally:
        _teardown()


def test_partial_batch_leaves_reduced_lambda_without_restore():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        strategy = tools.Strategy([0.0] * 5, sigma=1.0, offsprings=6)
        restart = tools.RestartStrategy(strategy, mode="ipop", budget=65)
        toolbox = Toolbox()
        toolbox.register("evaluate", tools.bm_sphere)
        toolbox.register("generate", restart.generate, creator.__dict__[IND])
        toolbox.register("update", restart.update)
        tools.ea_generate_update_restarts(toolbox, restart, log_restarts=False)
        assert restart.evals_used == 65
        assert strategy.lamb == 5
        assert not hasattr(restart, "_saved_lamb")
    finally:
        _teardown()

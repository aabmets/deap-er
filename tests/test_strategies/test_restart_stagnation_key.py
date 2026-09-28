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
from deap_er.private.strategies.restart_common import scalar_fitness

FIT = "RST_SK_FIT"
IND = "RST_SK_IND"


def _teardown():
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


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
        restart = tools.RestartStrategy(strategy, mode="ipop", budget=100, restart_centroid="best")
        restart.generate(creator.__dict__[IND])
        better = creator.__dict__[IND]([1.0, 1.0, 1.0])
        better.fitness.values = (10.0,)
        worse = creator.__dict__[IND]([0.0, 0.0, 0.0])
        worse.fitness.values = (5.0,)
        restart.update([worse, better])
        restart.restart()
        assert list(strategy.centroid) == better
    finally:
        _teardown()


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
            restart_centroid="best",
            stagnation_key=lambda ind: tools.hypervolume([ind], ref),
        )

        def evaluate(ind):
            return (ind[0] ** 2, ind[1] ** 2)

        offspring = restart.generate(creator.__dict__[IND])
        for ind in offspring:
            ind.fitness.values = evaluate(ind)
        restart.update(offspring)
        best = max(offspring, key=lambda ind: tools.hypervolume([ind], ref))
        restart.restart()
        assert [list(parent) for parent in mo.parents] == [list(best)] * 4
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
            restart_centroid="best",
            stagnation_key=lambda ind: float(ind.fitness.values[0]),
        )
        restart.generate(creator.__dict__[IND])
        better = creator.__dict__[IND]([1.0, 1.0, 1.0])
        better.fitness.values = (10.0,)
        worse = creator.__dict__[IND]([0.0, 0.0, 0.0])
        worse.fitness.values = (5.0,)
        restart.update([worse, better])
        restart.restart()
        assert list(strategy.centroid) == better
    finally:
        _teardown()


def test_stagnation_key_does_not_leak_into_best_fitness_or_target():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        strategy = tools.Strategy([0.0] * 2, sigma=1.0, offsprings=2)
        restart = tools.RestartStrategy(
            strategy,
            mode="ipop",
            budget=100,
            target_f=5.0,
            stagnation_key=lambda ind: -10.0 * ind.fitness.values[0],
        )
        good = creator.__dict__[IND]([1.0, 1.0])
        good.fitness.values = (2.0,)
        bad = creator.__dict__[IND]([2.0, 2.0])
        bad.fitness.values = (8.0,)
        restart.update([good, bad])
        assert restart.best_fitness == 2.0
        assert restart.is_done()
    finally:
        _teardown()

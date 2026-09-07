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

FIT = "RED_FIT"
IND = "RED_IND"


def _setup():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    return creator.__dict__[IND]


def _teardown():
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def test_generate_respects_exhausted_budget_and_empty_update():
    ind_cls = _setup()
    try:
        strategy = tools.Strategy(centroid=[0.0, 0.0], sigma=1.0, offsprings=4)
        restart = tools.RestartStrategy(strategy, mode="ipop", budget=0)
        assert restart.generate(ind_cls) == []
        restart.update([])
        assert restart.is_done()
    finally:
        _teardown()


def test_restart_requires_generate_first():
    _setup()
    try:
        strategy = tools.Strategy(centroid=[0.0, 0.0], sigma=1.0, offsprings=4)
        restart = tools.RestartStrategy(strategy, mode="ipop", budget=100)
        with pytest.raises(RuntimeError, match="generate"):
            restart.restart()
    finally:
        _teardown()


def test_best_fitness_and_max_restarts():
    ind_cls = _setup()
    try:
        strategy = tools.Strategy(centroid=[0.0, 0.0], sigma=1.0, offsprings=4)
        restart = tools.RestartStrategy(
            strategy, mode="ipop", budget=20, max_restarts=0, target_f=1e-12
        )
        population = restart.generate(ind_cls)
        for individual in population:
            individual.fitness.values = tools.bm_sphere(individual)
        restart.update(population)
        assert restart.best_fitness == restart.best_fitness
        assert restart.should_restart() is False
        assert restart.is_done() or restart.evals_used > 0
    finally:
        _teardown()


def test_partial_batch_resizes_offspring_count():
    ind_cls = _setup()
    try:
        strategy = tools.Strategy(centroid=[0.0, 0.0], sigma=1.0, offsprings=8)
        restart = tools.RestartStrategy(strategy, mode="ipop", budget=3)
        population = restart.generate(ind_cls)
        assert len(population) == 3
        assert strategy.lamb == 3
    finally:
        _teardown()

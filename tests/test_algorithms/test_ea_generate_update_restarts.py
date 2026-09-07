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
from deap_er import Fitness, Toolbox, creator, tools

FIT = "EA_RST_FIT"
IND = "EA_RST_IND"


def _setup(dim: int = 5):
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    strategy = tools.Strategy(centroid=[0.0] * dim, sigma=1.0, offsprings=10)
    toolbox = Toolbox()
    toolbox.register("evaluate", tools.bm_sphere)
    return strategy, toolbox


def _teardown():
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def test_sphere_convergence():
    tools.rng.seed(0)
    strategy, toolbox = _setup()
    try:
        restart = tools.RestartStrategy(strategy, mode="bipop", budget=50_000, sigma_large=1.0)
        toolbox.register("generate", restart.generate, creator.__dict__[IND])
        toolbox.register("update", restart.update)
        pop, logbook = tools.ea_generate_update_restarts(toolbox, restart)
        best = min(ind.fitness.values[0] for ind in pop)
        assert best < 1e-6
        assert "lambda" in logbook.header
    finally:
        _teardown()


def test_rastrigin_beats_plain_cma():
    dim = 10
    budget = 80_000
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        tools.rng.seed(42)
        strategy_plain = tools.Strategy(centroid=[4.8] * dim, sigma=2.0, offsprings=10)
        toolbox_plain = Toolbox()
        toolbox_plain.register("evaluate", tools.bm_rastrigin)
        toolbox_plain.register("generate", strategy_plain.generate, creator.__dict__[IND])
        toolbox_plain.register("update", strategy_plain.update)
        pop_plain, _ = tools.ea_generate_update(toolbox_plain, generations=budget // 10)
        plain_best = min(ind.fitness.values[0] for ind in pop_plain)

        tools.rng.seed(42)
        strategy_rst = tools.Strategy(centroid=[4.8] * dim, sigma=2.0, offsprings=10)
        toolbox_rst = Toolbox()
        toolbox_rst.register("evaluate", tools.bm_rastrigin)
        restart = tools.RestartStrategy(strategy_rst, mode="bipop", budget=budget)
        toolbox_rst.register("generate", restart.generate, creator.__dict__[IND])
        toolbox_rst.register("update", restart.update)
        pop_rst, _ = tools.ea_generate_update_restarts(toolbox_rst, restart, log_restarts=False)
        restart_best = min(ind.fitness.values[0] for ind in pop_rst)

        assert restart.restart_count >= 1
        assert restart_best < plain_best
    finally:
        _teardown()


def test_target_f_stops_early():
    strategy, toolbox = _setup()
    try:
        restart = tools.RestartStrategy(
            strategy,
            mode="ipop",
            budget=1_000_000,
            target_f=1e-3,
        )
        toolbox.register("generate", restart.generate, creator.__dict__[IND])
        toolbox.register("update", restart.update)
        tools.ea_generate_update_restarts(toolbox, restart, log_restarts=False)
        assert restart.best_fitness <= 1e-3
        assert restart.evals_used < 1_000_000
    finally:
        _teardown()

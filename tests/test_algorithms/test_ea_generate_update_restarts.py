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


def test_generate_update_restarts_uses_evaluate_batch_when_registered():
    tools.rng.seed(0)
    strategy, toolbox = _setup(dim=3)
    try:
        batches = []

        def evaluate_batch(individuals):
            batches.append(len(individuals))
            return [tools.bm_sphere(ind) for ind in individuals]

        def forbidden_map(*_args, **_kwargs):
            raise AssertionError("map must not be used while evaluate_batch is registered")

        restart = tools.RestartStrategy(strategy, mode="ipop", budget=12, sigma_large=1.0)
        toolbox.register("evaluate_batch", evaluate_batch)
        toolbox.register("map", forbidden_map)
        toolbox.register("generate", restart.generate, creator.__dict__[IND])
        toolbox.register("update", restart.update)
        _, logbook = tools.ea_generate_update_restarts(toolbox, restart, log_restarts=False)

        assert batches
        assert sum(batches) == sum(logbook.select("nevals"))
    finally:
        _teardown()


def test_sphere_convergence():
    tools.rng.seed(0)
    strategy, toolbox = _setup()
    try:
        restart = tools.RestartStrategy(strategy, mode="bipop", budget=50_000, sigma_large=1.0)
        toolbox.register("generate", restart.generate, creator.__dict__[IND])
        toolbox.register("update", restart.update)
        _, logbook = tools.ea_generate_update_restarts(toolbox, restart)
        assert restart.best_fitness < 1e-6
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
        tools.ea_generate_update_restarts(toolbox_rst, restart, log_restarts=False)

        assert restart.restart_count >= 1
        assert restart.best_fitness < plain_best
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


def test_empty_generate_keeps_last_evaluated_population():
    # generate() returning [] is the documented stop signal (budget
    # exhausted, or a caller that has no more samples). The last
    # evaluated batch must still be returned — not overwritten by [].
    tools.rng.seed(0)
    strategy, toolbox = _setup(dim=3)
    try:
        restart = tools.RestartStrategy(strategy, mode="ipop", budget=200, sigma_large=1.0)
        calls = {"n": 0}
        last_batch: list = []

        def generate():
            calls["n"] += 1
            if calls["n"] > 2:
                return []
            batch = restart.generate(creator.__dict__[IND])
            last_batch[:] = batch
            return batch

        toolbox.register("generate", generate)
        toolbox.register("update", restart.update)
        population, logbook = tools.ea_generate_update_restarts(
            toolbox, restart, log_restarts=False
        )

        assert calls["n"] == 3
        assert logbook.select("gen") == [1, 2]
        assert population is not last_batch
        assert population == last_batch
        assert len(population) > 0
        assert all(ind.fitness.is_valid() for ind in population)
    finally:
        _teardown()

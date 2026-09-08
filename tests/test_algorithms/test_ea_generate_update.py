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

FIT = "EA_GU_FIT"
IND = "EA_GU_IND"


def _setup(dim: int = 3, offsprings: int = 4):
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    strategy = tools.Strategy(centroid=[0.0] * dim, sigma=1.0, offsprings=offsprings)
    toolbox = Toolbox()
    toolbox.register("evaluate", tools.bm_sphere)
    return strategy, toolbox


def _teardown():
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def test_standard_cma():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    try:
        dimensions = 5
        strategy = tools.Strategy(centroid=[0.0] * dimensions, sigma=1.0)

        toolbox = Toolbox()
        toolbox.register("evaluate", tools.bm_sphere)
        toolbox.register("generate", strategy.generate, creator.__dict__[IND])
        toolbox.register("update", strategy.update)

        pop, _ = tools.ea_generate_update(toolbox, generations=100)
        (best,) = tools.sel_best(pop, sel_count=1)

        assert best.fitness.values < (1e-8,)
    finally:
        del creator.__dict__[FIT]
        del creator.__dict__[IND]


def test_empty_generate_keeps_last_evaluated_population():
    # generate() returning [] is a stop signal (no more samples).
    # The last evaluated batch must still be returned — not overwritten
    # by [] — and update must not be called with that empty list.
    tools.rng.seed(0)
    strategy, toolbox = _setup()
    try:
        calls = {"n": 0}
        last_batch = []
        updated = []

        def generate():
            calls["n"] += 1
            if calls["n"] > 2:
                return []
            batch = strategy.generate(creator.__dict__[IND])
            last_batch[:] = batch
            return batch

        def update(population):
            updated.append(list(population))
            strategy.update(population)

        toolbox.register("generate", generate)
        toolbox.register("update", update)
        population, logbook = tools.ea_generate_update(toolbox, generations=4)

        assert calls["n"] == 3
        assert logbook.select("gen") == [1, 2]
        assert len(population) == len(last_batch) > 0
        assert {id(ind) for ind in population} == {id(ind) for ind in last_batch}
        assert all(ind.fitness.is_valid() for ind in population)
        assert updated
        assert all(batch for batch in updated)
    finally:
        _teardown()


def test_empty_generate_from_the_start_returns_empty():
    tools.rng.seed(0)
    _strategy, toolbox = _setup()
    try:
        updates = []

        def generate():
            return []

        def update(population):
            updates.append(list(population))

        toolbox.register("generate", generate)
        toolbox.register("update", update)
        population, logbook = tools.ea_generate_update(toolbox, generations=3)

        assert population == []
        assert logbook.select("gen") == []
        assert updates == []
    finally:
        _teardown()


def test_empty_generate_does_not_call_cma_update():
    # Strategy.update ranks the batch and dots weights against the
    # top-mu rows. An empty list makes that product fail, so the
    # driver must not forward [] to the registered update.
    tools.rng.seed(0)
    strategy, toolbox = _setup()
    try:
        calls = {"n": 0}

        def generate():
            calls["n"] += 1
            if calls["n"] > 1:
                return []
            return strategy.generate(creator.__dict__[IND])

        toolbox.register("generate", generate)
        toolbox.register("update", strategy.update)
        population, logbook = tools.ea_generate_update(toolbox, generations=3)

        assert calls["n"] == 2
        assert logbook.select("gen") == [1]
        assert len(population) > 0
        assert all(ind.fitness.is_valid() for ind in population)
    finally:
        _teardown()

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

FIT = "RESAMPLE_FIT"
IND = "RESAMPLE_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    yield creator.__dict__[IND]
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def test_resample_writes_mean(ind_cls):
    individual = ind_cls([0])
    draws = iter([1.0, 3.0, 5.0])

    def evaluate(_ind):
        return (next(draws),)

    values = tools.resample(individual, evaluate, 3)

    assert values == (3.0,)
    assert individual.fitness.values == (3.0,)


def test_resample_without_write(ind_cls):
    individual = ind_cls([0])

    def evaluate(_ind):
        return (2.0,)

    values = tools.resample(individual, evaluate, 2, write=False)

    assert values == (2.0,)
    assert not individual.fitness.is_valid()


def test_resample_rejects_n_lt_1(ind_cls):
    individual = ind_cls([0])
    with pytest.raises(ValueError, match="n must"):
        tools.resample(individual, lambda _ind: (1.0,), 0)


def test_resample_aggregate_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        tools.resample_aggregate([(1.0,), (2.0, 3.0)])


def test_noisy_draw_key_distinct_per_draw():
    assert tools.noisy_draw_key("expr", 0) != tools.noisy_draw_key("expr", 1)


def test_resample_uses_eval_cache_per_draw(ind_cls):
    calls = []

    def evaluate(_ind):
        calls.append(1)
        return (float(len(calls)),)

    cache = tools.EvalCache(evaluate)
    individual = ind_cls([0])
    tools.resample(individual, evaluate, 3, cache=cache, key="expr")
    assert len(calls) == 3
    tools.resample(individual, evaluate, 3, cache=cache, key="expr")
    assert len(calls) == 3


def test_resample_same_draw_hits_cache(ind_cls):
    calls = []

    def evaluate(_ind):
        calls.append(1)
        return (1.0,)

    cache = tools.EvalCache(evaluate)
    individual = ind_cls([0])
    draw_key = tools.noisy_draw_key("expr", 0)
    cache.evaluate(individual, key=draw_key)
    cache.evaluate(individual, key=draw_key)
    assert calls == [1]


def test_race_stop_empty_input():
    result = tools.race_stop([], lambda _ind: (0.0,), 3)
    assert result.survivors == []
    assert result.nevals == 0
    assert result.rounds == 0


def test_race_eval_charge():
    assert tools.race_eval_charge(5, 2) == 10


def test_race_stop_eliminates_worse(ind_cls):
    population = [ind_cls([rank]) for rank in range(4)]

    def evaluate(ind):
        base = float(ind[0] * 10)
        return (base + tools.rng.random() * 0.01,)

    tools.rng.seed(0)
    result = tools.race_stop(population, evaluate, 12, min_survivors=1)

    assert len(result.survivors) == 1
    assert result.survivors[0][0] == 0


def test_race_stop_eliminates_worse_deterministic(ind_cls):
    population = [ind_cls([rank]) for rank in range(4)]

    def evaluate(ind):
        return (float(ind[0]),)

    result = tools.race_stop(population, evaluate, 4, min_survivors=1)

    assert len(result.survivors) == 1
    assert result.survivors[0][0] == 0
    assert result.nevals == 8
    assert result.rounds == 2


def test_race_stop_respects_min_survivors(ind_cls):
    population = [ind_cls([rank]) for rank in range(4)]

    def evaluate(ind):
        return (float(ind[0]),)

    result = tools.race_stop(population, evaluate, 10, min_survivors=2)

    assert len(result.survivors) >= 2


def test_race_stop_with_cache(ind_cls):
    calls = []

    def evaluate(ind):
        calls.append(ind[0])
        return (float(ind[0]),)

    cache = tools.EvalCache(evaluate)
    population = [ind_cls([0]), ind_cls([1])]
    tools.race_stop(
        population,
        evaluate,
        2,
        cache=cache,
        key_fn=lambda ind: ind[0],
        min_survivors=1,
    )
    assert calls == [0, 1, 0, 1]
    tools.race_stop(
        population,
        evaluate,
        2,
        cache=cache,
        key_fn=lambda ind: ind[0],
        min_survivors=1,
    )
    assert calls == [0, 1, 0, 1]


def test_race_stop_write_assigns_aggregate(ind_cls):
    winner = ind_cls([0])
    loser = ind_cls([5])
    draws = {0: iter([1.0, 3.0]), 5: iter([50.0, 50.0])}

    def evaluate(ind):
        return (next(draws[ind[0]]),)

    result = tools.race_stop([winner, loser], evaluate, 2, write=True, min_survivors=1)

    assert result.survivors == [winner]
    assert winner.fitness.values == (2.0,)

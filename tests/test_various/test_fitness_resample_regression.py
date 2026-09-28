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
"""Permanent regression tests for noisy fitness resample (roadmap item 40)."""

import numpy
import pytest
from deap_er import Fitness, creator, tools
from deap_er.private.various.race_rounds import race_z_score

MAX_FIT = "RESAMPLE_REG_MAX_FIT"
MAX_IND = "RESAMPLE_REG_MAX_IND"
CACHE_FIT = "RESAMPLE_REG_CACHE_FIT"
CACHE_IND = "RESAMPLE_REG_CACHE_IND"


@pytest.fixture
def max_ind_cls():
    creator.create_type(MAX_FIT, Fitness, weights=(1.0,))
    creator.create_type(MAX_IND, list, fitness=creator.__dict__[MAX_FIT])
    yield creator.__dict__[MAX_IND]
    del creator.__dict__[MAX_FIT]
    del creator.__dict__[MAX_IND]


@pytest.fixture
def cache_ind_cls():
    creator.create_type(CACHE_FIT, Fitness, weights=(-1.0,))
    creator.create_type(CACHE_IND, list, fitness=creator.__dict__[CACHE_FIT])
    yield creator.__dict__[CACHE_IND]
    del creator.__dict__[CACHE_FIT]
    del creator.__dict__[CACHE_IND]


def test_race_stop_honors_maximize_weights_before_evaluation(max_ind_cls):
    """Unevaluated maximize fitness must not default to minimization."""
    population = [max_ind_cls([rank]) for rank in range(4)]

    def evaluate(ind):
        return (float(ind[0]),)

    result = tools.race_stop(population, evaluate, 4, min_survivors=1)

    assert len(result.survivors) == 1
    assert result.survivors[0][0] == 3


def test_race_stop_cache_uses_distinct_draw_keys_without_key_fn(cache_ind_cls):
    """Each racing round must miss the cache when key_fn is omitted."""
    calls = []

    def evaluate(_ind):
        calls.append(1)
        return (float(len(calls)),)

    cache = tools.EvalCache(evaluate)
    population = [cache_ind_cls([rank]) for rank in range(3)]
    result = tools.race_stop(population, evaluate, 3, cache=cache, min_survivors=1)

    assert result.rounds == 3
    assert len(calls) == 9


def test_resample_cache_without_key_keeps_individuals_apart(cache_ind_cls):
    """Without ``key`` the draw entries must not be shared across genomes."""
    cache = tools.EvalCache(lambda ind: (float(ind[0]),))

    first = tools.resample(cache_ind_cls([1]), cache.evaluate, 2, cache=cache)
    second = tools.resample(cache_ind_cls([7]), cache.evaluate, 2, cache=cache)

    assert first == (1.0,)
    assert second == (7.0,)


def test_race_stop_cache_without_key_fn_sees_in_place_mutation(max_ind_cls):
    """The default cache key follows the genome, not ``id(individual)``."""
    cache = tools.EvalCache(lambda ind: (float(ind[0]),))
    individual = max_ind_cls([1])
    tools.race_stop([individual], cache.evaluate, 1, cache=cache, write=True)
    individual[0] = 7
    del individual.fitness.values

    tools.race_stop([individual], cache.evaluate, 1, cache=cache, write=True)

    assert individual.fitness.values == (7.0,)


def test_race_stop_restores_min_survivors_for_ndarray_individuals():
    """Restoring survivors must not compare ndarray genomes with ``==``."""
    creator.create_type("RACE_NP_FIT", Fitness, weights=(1.0,))
    creator.create_type("RACE_NP_IND", numpy.ndarray, fitness=creator.__dict__["RACE_NP_FIT"])
    try:
        ind_cls = creator.__dict__["RACE_NP_IND"]
        population = [ind_cls([float(rank), 1.0]) for rank in range(4)]

        result = tools.race_stop(population, lambda ind: (float(ind[0]),), 3, min_survivors=3)
    finally:
        del creator.__dict__["RACE_NP_FIT"]
        del creator.__dict__["RACE_NP_IND"]

    assert [float(ind[0]) for ind in result.survivors] == [3.0, 2.0, 1.0]


def test_race_stop_restores_equal_but_distinct_individuals(max_ind_cls):
    """An equal genome held by a different object is still a distinct survivor."""
    leader, twin = max_ind_cls([1]), max_ind_cls([1])
    other, last = max_ind_cls([2]), max_ind_cls([3])
    scores = {id(leader): 10.0, id(other): 5.0, id(twin): 0.0, id(last): -1.0}

    result = tools.race_stop(
        [leader, twin, other, last], lambda ind: (scores[id(ind)],), 3, min_survivors=3
    )

    assert [id(ind) for ind in result.survivors] == [id(leader), id(other), id(twin)]


def test_race_z_score_follows_alpha():
    """Every alpha maps to its own two-sided critical value."""
    assert race_z_score(0.05) == pytest.approx(1.959964, abs=1e-6)
    assert race_z_score(0.2) == pytest.approx(1.281552, abs=1e-6)
    assert race_z_score(0.2) < race_z_score(0.1) < race_z_score(0.05) < race_z_score(0.01)


def test_race_stop_rejects_bad_alpha_before_racing(max_ind_cls):
    """An invalid alpha fails up front, even when no elimination round runs."""
    population = [max_ind_cls([0])]
    with pytest.raises(ValueError, match="alpha"):
        tools.race_stop(population, lambda _ind: (0.0,), 1, alpha=1.5)

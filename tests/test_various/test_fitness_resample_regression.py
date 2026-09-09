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

import pytest
from deap_er import Fitness, creator, tools

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

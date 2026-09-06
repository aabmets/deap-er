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
from collections import Counter

import pytest
from deap_er import Fitness, creator, tools


def test_stochastic_universal_sampling_zero_count(single_obj, make):
    population = [make(single_obj, [i], (float(i + 1),)) for i in range(6)]

    assert tools.sel_stochastic_universal_sampling(population, 0) == []


def test_proportionate_selection_empty_or_zero_count_returns_empty(single_obj, make):
    population = [make(single_obj, [0], (1.0,))]
    assert tools.sel_roulette([], 0) == []
    assert tools.sel_roulette([], 3) == []
    assert tools.sel_roulette(population, 0) == []
    assert tools.sel_stochastic_universal_sampling([], 3) == []
    assert tools.sel_stochastic_universal_sampling(population, 0) == []


def test_stochastic_universal_sampling_returns_requested_count(single_obj, make):
    population = [make(single_obj, [i], (float(i + 1),)) for i in range(6)]

    chosen = tools.sel_stochastic_universal_sampling(population, 4)

    assert len(chosen) == 4
    assert all(ind in population for ind in chosen)


def test_roulette_returns_requested_count(single_obj, make):
    population = [make(single_obj, [i], (float(i + 1),)) for i in range(6)]

    tools.rng.seed(12)
    chosen = tools.sel_roulette(population, 5)

    assert len(chosen) == 5
    assert all(ind in population for ind in chosen)


def test_roulette_all_zero_fitness_returns_requested_count(single_obj, make):
    population = [make(single_obj, [i], (0.0,)) for i in range(5)]

    tools.rng.seed(12)
    chosen = tools.sel_roulette(population, 3)

    assert len(chosen) == 3
    assert all(ind in population for ind in chosen)


@pytest.mark.parametrize(
    "selector, sel_count",
    [
        (tools.sel_roulette, 200),
        (tools.sel_stochastic_universal_sampling, 30),
    ],
)
def test_proportionate_selection_prefers_best_when_minimizing(selector, sel_count, make):
    creator.create_type("SEL_MIN_FIT", Fitness, weights=(-1.0,))
    creator.create_type("SEL_MIN_IND", list, fitness=creator.__dict__["SEL_MIN_FIT"])
    try:
        values = (1.0, 2.0, 10.0)
        population = [make(creator.__dict__["SEL_MIN_IND"], [value], (value,)) for value in values]
        tools.rng.seed(0)
        chosen = selector(population, sel_count)
        counts = Counter(ind[0] for ind in chosen)
        assert len(chosen) == sel_count
        assert counts[1.0] > counts[10.0]
    finally:
        del creator.__dict__["SEL_MIN_FIT"]
        del creator.__dict__["SEL_MIN_IND"]

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
from deap_er import tools


def test_stochastic_universal_sampling_zero_count(single_obj, make):
    population = [make(single_obj, [i], (float(i + 1),)) for i in range(6)]

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

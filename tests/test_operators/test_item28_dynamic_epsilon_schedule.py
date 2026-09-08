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
from typing import Any

import pytest
from deap_er import Fitness, creator, tools

FIT = "ITEM28_FIT"
IND = "ITEM28_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(FIT, Fitness, weights=(-1.0,) * 8)
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    yield creator.__dict__[IND]
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def _make(ind_cls, genes, values):
    individual = ind_cls(genes)
    individual.fitness.values = values
    return individual


def test_epsilon_static_uses_population_mad(multi_obj, make):
    values = [(10.0, 0.0), (9.0, 50.0), (8.0, 100.0), (7.0, 150.0)]
    population = [make(multi_obj, [i], value) for i, value in enumerate(values)]

    tools.rng.seed(11)
    static = tools.sel_epsilon_lexicase(population, 1, mode="epsilon_static")
    tools.rng.seed(11)
    auto = tools.sel_epsilon_lexicase(population, 1)
    tools.rng.seed(11)
    dynamic = tools.sel_epsilon_lexicase(population, 1, mode="epsilon_dynamic")

    assert static == auto
    assert dynamic != static


def test_epsilon_semi_differs_from_static_on_shrunk_pool(multi_obj, make):
    values = [(10.0, 0.0), (9.0, 50.0), (8.0, 100.0), (7.0, 150.0)]
    population = [make(multi_obj, [i], value) for i, value in enumerate(values)]

    tools.rng.seed(17)
    static = tools.sel_epsilon_lexicase(population, 1, mode="epsilon_static")
    tools.rng.seed(17)
    semi = tools.sel_epsilon_lexicase(population, 1, mode="epsilon_semi")

    assert semi != static


def test_epsilon_dynamic_differs_from_static(multi_obj, make):
    values = [(10.0, 0.0), (9.0, 50.0), (8.0, 100.0), (7.0, 150.0)]
    population = [make(multi_obj, [i], value) for i, value in enumerate(values)]

    tools.rng.seed(17)
    static = tools.sel_epsilon_lexicase(population, 1, mode="epsilon_static")
    tools.rng.seed(17)
    dynamic = tools.sel_epsilon_lexicase(population, 1, mode="epsilon_dynamic")

    assert dynamic != static


def test_epsilon_modes_accept_matrix(multi_obj, make):
    values = [(10.0, 0.0), (9.0, 50.0), (8.0, 100.0), (7.0, 150.0)]
    population = [make(multi_obj, [i], value) for i, value in enumerate(values)]
    matrix = tools.fitness_case_matrix(population)

    for mode in ("epsilon_static", "epsilon_semi", "epsilon_dynamic"):
        tools.rng.seed(3)
        default = tools.sel_epsilon_lexicase(population, 4, mode=mode)
        tools.rng.seed(3)
        packed = tools.sel_epsilon_lexicase(population, 4, mode=mode, matrix=matrix)
        assert default == packed


def test_downsample_random_respects_case_count(ind_cls):
    population = [_make(ind_cls, [i], (float(i),) * 8) for i in range(4)]

    for seed in range(10):
        tools.rng.seed(seed)
        subset = tools.next_downsample_cases(population, 3, 0, mode="random")
        assert len(subset) == len(set(subset)) == 3


def test_downsample_informed_matches_helper(ind_cls):
    population = [
        _make(ind_cls, [0], (0.0, 1.0, 1.0, 0.0) + (1.0,) * 4),
        _make(ind_cls, [1], (1.0, 0.0, 1.0, 1.0) + (1.0,) * 4),
        _make(ind_cls, [2], (1.0, 1.0, 0.0, 1.0) + (1.0,) * 4),
    ]

    tools.rng.seed(5)
    scheduled = tools.next_downsample_cases(population, 2, 0, mode="informed")
    tools.rng.seed(5)
    direct = tools.sample_informed_cases(population, 2)

    assert scheduled == direct


def test_downsample_cohort_fixed_and_rotating(ind_cls):
    population = [_make(ind_cls, [0], (0.0,) * 8)]

    fixed = tools.next_downsample_cases(population, 2, 0, mode="cohort", cohort=[1, 3, 4])
    assert fixed == [1, 3]

    rotating = [
        tools.next_downsample_cases(
            population,
            2,
            gen,
            mode="cohort",
            cohorts=[[0, 1], [2, 3], [4, 0]],
        )
        for gen in range(4)
    ]
    assert rotating == [[0, 1], [2, 3], [4, 0], [0, 1]]


def test_downsample_held_out_rotates(ind_cls):
    population = [_make(ind_cls, [0], (0.0,) * 8)]
    held = tools.CaseExam.from_cases([0, 1, 2, 3, 4, 5], 8)

    gen0 = tools.next_downsample_cases(population, 2, 0, mode="held_out", held_out=held)
    gen1 = tools.next_downsample_cases(population, 2, 1, mode="held_out", held_out=held)
    gen2 = tools.next_downsample_cases(population, 2, 2, mode="held_out", held_out=held)
    gen3 = tools.next_downsample_cases(population, 2, 3, mode="held_out", held_out=held)

    assert gen0 == [0, 1]
    assert gen1 == [2, 3]
    assert gen2 == [4, 5]
    assert gen3 == [0, 1]


def test_downsample_empty_count_or_pool(ind_cls):
    population = [_make(ind_cls, [0], (0.0, 1.0, 0.0, 1.0) + (0.0,) * 4)]

    assert tools.next_downsample_cases(population, 0, 0) == []
    with pytest.raises(ValueError, match="non-empty"):
        tools.next_downsample_cases([], 2, 0)
    bad_count: Any = "2"
    with pytest.raises(ValueError, match="must be an int"):
        tools.next_downsample_cases(population, bad_count, 0)


def test_downsample_mode_requires_arguments(ind_cls):
    population = [_make(ind_cls, [0], (0.0, 1.0, 0.0, 1.0) + (0.0,) * 4)]

    with pytest.raises(ValueError, match="cohort"):
        tools.next_downsample_cases(population, 2, 0, mode="cohort")
    with pytest.raises(ValueError, match="held_out"):
        tools.next_downsample_cases(population, 2, 0, mode="held_out")


def test_downsample_cohort_rejects_out_of_range(ind_cls):
    population = [_make(ind_cls, [0], (0.0, 1.0, 0.0, 1.0) + (0.0,) * 4)]

    with pytest.raises(IndexError, match="case index 9"):
        tools.next_downsample_cases(population, 2, 0, mode="cohort", cohort=[0, 9])

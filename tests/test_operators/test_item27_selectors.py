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

import numpy
import pytest
from deap_er import Fitness, creator, tools

CASE_FIT = "ITEM27_CASE_FIT"
CASE_IND = "ITEM27_CASE_IND"


@pytest.fixture
def case_types():
    creator.create_type(CASE_FIT, Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
    creator.create_type(CASE_IND, list, fitness=creator.__dict__[CASE_FIT])
    yield creator.__dict__[CASE_IND]
    del creator.__dict__[CASE_FIT]
    del creator.__dict__[CASE_IND]


def test_batch_epsilon_lexicase_fewer_batches_than_cases(case_types, make):
    population = [
        make(case_types, [0], (1.0, 9.0, 9.0, 9.0)),
        make(case_types, [1], (9.0, 1.0, 9.0, 9.0)),
        make(case_types, [2], (9.0, 9.0, 1.0, 9.0)),
        make(case_types, [3], (9.0, 9.0, 9.0, 1.0)),
    ]
    matrix = tools.fitness_case_matrix(population)

    tools.rng.seed(0)
    chosen = tools.sel_batch_epsilon_lexicase(population, 8, batch_size=2, matrix=matrix)

    assert len(chosen) == 8
    assert all(ind in population for ind in chosen)


def test_batch_epsilon_lexicase_re_batches_each_selection(case_types, make):
    values = [(10.0, 0.0, 5.0, 0.0), (9.0, 50.0, 8.0, 100.0), (8.0, 100.0, 7.0, 150.0)]
    population = [make(case_types, [i], value) for i, value in enumerate(values)]
    matrix = tools.fitness_case_matrix(population)

    tools.rng.seed(21)
    batched = tools.sel_batch_epsilon_lexicase(population, 6, batch_size=2, matrix=matrix)
    tools.rng.seed(21)
    one_at_a_time = [
        tools.sel_batch_epsilon_lexicase(population, 1, batch_size=2, matrix=matrix)[0]
        for _ in range(6)
    ]

    assert batched == one_at_a_time


def test_batch_epsilon_lexicase_mean_reduction_matches_subset_lexicase(case_types, make):
    best = make(case_types, [0], (0.0, 10.0, 10.0, 10.0))
    population = [best, make(case_types, [1], (10.0, 0.0, 0.0, 0.0))]

    chosen = tools.sel_batch_epsilon_lexicase(
        population,
        10,
        batch_size=4,
        epsilon=0.0,
        cases=[0],
        reduction=tools.reduce_case_mean,
    )

    assert all(ind is best for ind in chosen)


def test_batch_epsilon_lexicase_respects_cases_subset(case_types, make):
    specialist = make(case_types, [0], (0.0, 10.0, 10.0, 10.0))
    population = [specialist, make(case_types, [1], (10.0, 0.0, 0.0, 0.0))]

    chosen = tools.sel_batch_epsilon_lexicase(
        population,
        10,
        batch_size=2,
        epsilon=0.0,
        cases=[0],
        reduction=tools.reduce_case_mean,
    )

    assert all(ind is specialist for ind in chosen)


def test_batch_epsilon_lexicase_invalid_batch_size(case_types, make):
    population = [make(case_types, [0], (1.0, 2.0, 3.0, 4.0))]

    with pytest.raises(ValueError, match="batch_size"):
        tools.sel_batch_epsilon_lexicase(population, 1, batch_size=0)


def test_batch_epsilon_lexicase_empty_pool_with_zero_count():
    assert tools.sel_batch_epsilon_lexicase([], 0, batch_size=2) == []


def test_batch_epsilon_lexicase_empty_pool_raises(case_types):
    with pytest.raises(IndexError):
        tools.sel_batch_epsilon_lexicase([], 1, batch_size=2)


def test_partition_case_batches_covers_every_case_once():
    tools.rng.seed(3)
    batches = tools.partition_case_batches([0, 1, 2, 3, 4], 2)

    assert sorted(idx for batch in batches for idx in batch) == [0, 1, 2, 3, 4]
    assert all(len(batch) <= 2 for batch in batches)


def test_batch_case_matrix_mse_shape(case_types, make):
    population = [
        make(case_types, [0], (1.0, 2.0, 3.0, 4.0)),
        make(case_types, [1], (5.0, 6.0, 7.0, 8.0)),
    ]
    matrix = tools.fitness_case_matrix(population)
    tools.rng.seed(0)
    reduced, weights = tools.batch_case_matrix(
        matrix,
        [0, 1, 2, 3],
        population[0].fitness.weights,
        2,
        tools.reduce_case_mse,
    )

    assert reduced.shape == (2, 2)
    assert len(weights) == 2
    assert weights == (-1.0, -1.0)


def test_tournament_cases_prefers_lower_mean_on_subset(case_types, make):
    best = make(case_types, [0], (1.0, 1.0, 9.0, 9.0))
    worse = make(case_types, [1], (9.0, 9.0, 1.0, 1.0))
    population = [worse, best]
    tools.rng.seed(1)

    chosen = tools.sel_tournament_cases(
        population,
        rounds=40,
        contestants=2,
        cases=[0, 1],
    )

    assert chosen.count(best) > chosen.count(worse)


def test_tournament_cases_with_informed_subset(case_types, make):
    values = [(0.0, 0.0, 1.0, 0.0), (1.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.0)]
    population = [make(case_types, [i], value) for i, value in enumerate(values)]
    matrix = tools.fitness_case_matrix(population)
    cases = tools.sample_informed_cases(population, 2, matrix=matrix)

    chosen = tools.sel_tournament_cases(
        population,
        rounds=5,
        contestants=2,
        cases=cases,
        matrix=matrix,
    )

    assert len(chosen) == 5
    assert all(ind in population for ind in chosen)


def test_tournament_cases_random_case_count(case_types, make):
    population = [
        make(case_types, [0], (1.0, 2.0, 3.0, 4.0)),
        make(case_types, [1], (4.0, 3.0, 2.0, 1.0)),
    ]

    tools.rng.seed(11)
    chosen = tools.sel_tournament_cases(population, rounds=4, contestants=2, case_count=2)

    assert len(chosen) == 4


def test_tournament_cases_custom_reduction(case_types, make):
    best = make(case_types, [0], (1.0, 9.0, 9.0, 9.0))
    worse = make(case_types, [1], (2.0, 1.0, 1.0, 1.0))
    population = [worse, best]
    tools.rng.seed(2)

    def _max_col(block: numpy.ndarray) -> numpy.ndarray:
        return numpy.max(block, axis=1)

    chosen = tools.sel_tournament_cases(
        population,
        rounds=40,
        contestants=2,
        cases=[0],
        reduction=_max_col,
    )

    assert chosen.count(best) > chosen.count(worse)


def test_tournament_cases_empty_rounds_or_pool(case_types, make):
    population = [make(case_types, [0], (1.0, 2.0, 3.0, 4.0))]

    assert tools.sel_tournament_cases(population, rounds=0, contestants=2) == []
    assert tools.sel_tournament_cases([], rounds=0, contestants=2) == []


def test_tournament_cases_empty_pool_raises():
    with pytest.raises(IndexError):
        tools.sel_tournament_cases([], rounds=1, contestants=2)


def test_tournament_cases_rejects_non_positive_contestants(case_types, make):
    population = [make(case_types, [0], (1.0, 2.0, 3.0, 4.0))]

    with pytest.raises(ValueError, match="contestants"):
        tools.sel_tournament_cases(population, rounds=1, contestants=0)


def test_tournament_cases_matrix_matches_default(case_types, make):
    population = [
        make(case_types, [0], (1.0, 2.0, 3.0, 4.0)),
        make(case_types, [1], (4.0, 3.0, 2.0, 1.0)),
    ]
    matrix = tools.fitness_case_matrix(population)

    for seed in range(10):
        tools.rng.seed(seed)
        default = tools.sel_tournament_cases(population, 6, 2, cases=[0, 1])
        tools.rng.seed(seed)
        packed = tools.sel_tournament_cases(population, 6, 2, cases=[0, 1], matrix=matrix)
        assert default == packed


def test_tournament_cases_invalid_case_count(case_types, make):
    population = [make(case_types, [0], (1.0, 2.0, 3.0, 4.0))]
    bad_count: Any = "2"

    with pytest.raises(ValueError, match="case_count"):
        tools.sel_tournament_cases(population, 1, 2, case_count=bad_count)

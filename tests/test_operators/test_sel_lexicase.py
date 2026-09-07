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

MANY_FIT = "LEX_MANY_FIT"
MANY_IND = "LEX_MANY_IND"


@pytest.fixture
def error_cases():
    creator.create_type(MANY_FIT, Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
    creator.create_type(MANY_IND, list, fitness=creator.__dict__[MANY_FIT])
    yield creator.__dict__[MANY_IND]
    del creator.__dict__[MANY_FIT]
    del creator.__dict__[MANY_IND]


def test_epsilon_lexicase_zero_epsilon_is_strict(single_obj, make):
    # An explicit epsilon of 0.0 must not be treated as "compute it from the MAD".
    best = make(single_obj, [0], (10.0,))
    population = [best] + [make(single_obj, [i], (v,)) for i, v in enumerate((9.0, 8.0, 0.0), 1)]

    chosen = tools.sel_epsilon_lexicase(population, 20, epsilon=0.0)

    assert all(ind is best for ind in chosen)


def test_epsilon_lexicase_recomputes_epsilon_for_each_selection(multi_obj, make):
    # The two cases have very different spreads, so reusing the epsilon of one
    # case while filtering on the other changes which candidates survive.
    values = [(10.0, 0.0), (9.0, 50.0), (8.0, 100.0), (7.0, 150.0)]
    population = [make(multi_obj, [i], value) for i, value in enumerate(values)]

    tools.rng.seed(99)
    batched = tools.sel_epsilon_lexicase(population, 8)
    tools.rng.seed(99)
    one_at_a_time = [tools.sel_epsilon_lexicase(population, 1)[0] for _ in range(8)]

    # Each selection must start from a fresh epsilon, so a batch of eight has to
    # match eight independent single selections drawn from the same seed.
    assert batched == one_at_a_time


def test_lexicase_cases_omit_held_out_specialist(multi_obj, make):
    first = make(multi_obj, [0], (10.0, 0.0))
    population = [first, make(multi_obj, [1], (0.0, 10.0))]

    chosen = tools.sel_lexicase(population, 20, cases=[0])

    assert all(ind is first for ind in chosen)


def test_lexicase_does_not_mutate_caller_cases(multi_obj, make):
    population = [make(multi_obj, [0], (10.0, 0.0)), make(multi_obj, [1], (0.0, 10.0))]
    cases = [0, 1]

    tools.sel_lexicase(population, 8, cases=cases)

    assert cases == [0, 1]


def test_lexicase_empty_cases_draws_from_the_pool(multi_obj, make):
    population = [make(multi_obj, [0], (10.0, 0.0)), make(multi_obj, [1], (0.0, 10.0))]

    chosen = tools.sel_lexicase(population, 10, cases=[])

    assert all(ind in population for ind in chosen)


def test_epsilon_lexicase_zero_epsilon_respects_cases(multi_obj, make):
    first = make(multi_obj, [0], (10.0, 0.0))
    population = [first, make(multi_obj, [1], (0.0, 10.0))]

    chosen = tools.sel_epsilon_lexicase(population, 20, 0.0, cases=[0])

    assert all(ind is first for ind in chosen)


def test_lexicase_out_of_range_case_raises(multi_obj, make):
    population = [make(multi_obj, [0], (1.0, 2.0))]

    with pytest.raises(IndexError, match="case index 9"):
        tools.sel_lexicase(population, 1, cases=[0, 9])


def test_informed_cases_prefers_distinct_solve_vectors(error_cases, make):
    # Cases 0, 1, and 3 are synonymous; case 2 is solved by a different subset.
    values = [(0.0, 0.0, 1.0, 0.0), (1.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.0)]
    population = [make(error_cases, [i], value) for i, value in enumerate(values)]

    for seed in range(20):
        tools.rng.seed(seed)
        subset = tools.sample_informed_cases(population, 2)
        assert 2 in subset
        assert len(subset) == len(set(subset)) == 2


def test_informed_cases_caps_at_available_indices(error_cases, make):
    values = [(0.0, 1.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0), (1.0, 1.0, 0.0, 1.0)]
    population = [make(error_cases, [i], value) for i, value in enumerate(values)]

    subset = tools.sample_informed_cases(population, 40)

    assert sorted(subset) == [0, 1, 2, 3]


def test_informed_cases_empty_count_or_pool(error_cases, make):
    population = [make(error_cases, [0], (0.0, 1.0, 1.0, 1.0))]

    assert tools.sample_informed_cases(population, 0) == []
    assert tools.sample_informed_cases(population, -3) == []
    with pytest.raises(ValueError, match="non-empty"):
        tools.sample_informed_cases([], 2)
    bad_count: Any = "2"
    with pytest.raises(ValueError, match="must be an int"):
        tools.sample_informed_cases(population, bad_count)


def test_informed_cases_invalid_first_fitness_raises(error_cases, make):
    # An unevaluated first individual must not collapse to an empty subset.
    invalid = error_cases([0])
    valid = make(error_cases, [1], (0.0, 1.0, 1.0, 1.0))

    with pytest.raises(ValueError, match="valid fitness"):
        tools.sample_informed_cases([invalid, valid], 2)


def test_lexicase_zero_count_on_empty_pool_returns_empty():
    assert tools.sel_lexicase([], 0) == []
    assert tools.sel_epsilon_lexicase([], 0) == []


def test_informed_cases_accepts_numpy_int_count(error_cases, make):
    values = [(0.0, 1.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0), (1.0, 1.0, 0.0, 1.0)]
    population = [make(error_cases, [i], value) for i, value in enumerate(values)]

    count: Any = numpy.int64(2)
    subset = tools.sample_informed_cases(population, count)

    assert len(subset) == 2


def test_lexicase_float_case_index_raises_index_error(multi_obj, make):
    population = [make(multi_obj, [0], (1.0, 2.0))]

    bad_cases: Any = [1.0]
    with pytest.raises(IndexError, match="case index"):
        tools.sel_lexicase(population, 1, cases=bad_cases)


def test_fitness_case_matrix_round_trip(multi_obj, make):
    population = [
        make(multi_obj, [0], (1.0, 2.0)),
        make(multi_obj, [1], (3.0, 4.0)),
    ]

    matrix = tools.fitness_case_matrix(population)

    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 1.0
    assert matrix[1, 1] == 4.0


def test_lexicase_matrix_matches_default_path(multi_obj, make):
    values = [(10.0, 0.0), (9.0, 50.0), (8.0, 100.0), (7.0, 150.0)]
    population = [make(multi_obj, [i], value) for i, value in enumerate(values)]
    matrix = tools.fitness_case_matrix(population)

    for seed in range(20):
        tools.rng.seed(seed)
        default = tools.sel_lexicase(population, 8)
        tools.rng.seed(seed)
        packed = tools.sel_lexicase(population, 8, matrix=matrix)
        assert default == packed


def test_epsilon_lexicase_matrix_matches_default_path(multi_obj, make):
    values = [(10.0, 0.0), (9.0, 50.0), (8.0, 100.0), (7.0, 150.0)]
    population = [make(multi_obj, [i], value) for i, value in enumerate(values)]
    matrix = tools.fitness_case_matrix(population)

    for seed in range(20):
        tools.rng.seed(seed)
        default = tools.sel_epsilon_lexicase(population, 8)
        tools.rng.seed(seed)
        packed = tools.sel_epsilon_lexicase(population, 8, matrix=matrix)
        assert default == packed


def test_lexicase_matrix_wrong_shape_raises(multi_obj, make):
    population = [make(multi_obj, [0], (1.0, 2.0))]

    with pytest.raises(ValueError, match="shape"):
        tools.sel_lexicase(population, 1, matrix=numpy.zeros((1, 3)))


def test_lexicase_matrix_mismatch_raises(multi_obj, make):
    population = [make(multi_obj, [0], (1.0, 2.0))]
    matrix = numpy.array([[9.0, 8.0]])

    with pytest.raises(ValueError, match="does not match"):
        tools.sel_lexicase(population, 1, matrix=matrix)


def test_informed_cases_matrix_matches_default(error_cases, make):
    values = [(0.0, 1.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0), (1.0, 1.0, 0.0, 1.0)]
    population = [make(error_cases, [i], value) for i, value in enumerate(values)]
    matrix = tools.fitness_case_matrix(population)

    for seed in range(20):
        tools.rng.seed(seed)
        default = tools.sample_informed_cases(population, 2)
        tools.rng.seed(seed)
        packed = tools.sample_informed_cases(population, 2, matrix=matrix)
        assert default == packed


def test_lexicase_unevaluated_fitness_random_draw(single_obj):
    """Unevaluated individuals (zero cases) draw from the pool without error."""
    ind = single_obj([0])
    population = [ind]

    tools.rng.seed(7)
    chosen = tools.sel_lexicase(population, 3)

    assert len(chosen) == 3
    assert all(item is ind for item in chosen)


def test_epsilon_lexicase_unevaluated_fitness_random_draw(single_obj):
    ind = single_obj([0])
    population = [ind]

    tools.rng.seed(7)
    chosen = tools.sel_epsilon_lexicase(population, 3)

    assert len(chosen) == 3
    assert all(item is ind for item in chosen)


def test_lexicase_empty_pool_with_positive_sel_count_raises_index_error():
    with pytest.raises(IndexError):
        tools.sel_lexicase([], 1)


def test_epsilon_lexicase_empty_pool_with_positive_sel_count_raises_index_error():
    with pytest.raises(IndexError):
        tools.sel_epsilon_lexicase([], 1)


def test_lexicase_nan_fitness_falls_back_to_pool(multi_obj, make):
    nan_ind = make(multi_obj, [0], (float("nan"), 0.0))
    other = make(multi_obj, [1], (1.0, 0.0))
    population = [nan_ind, other]

    tools.rng.seed(0)
    chosen = tools.sel_lexicase(population, 5)

    assert len(chosen) == 5
    assert all(ind in population for ind in chosen)


def test_lexicase_trust_matrix_skips_value_check(multi_obj, make, monkeypatch):
    population = [make(multi_obj, [0], (1.0, 2.0))]
    matrix = numpy.array([[9.0, 8.0]])

    calls: list[str] = []

    def _spy_pack(*args, **kwargs):
        calls.append("pack")
        raise AssertionError("fitness_case_matrix should not run during trust validation")

    monkeypatch.setattr(
        "deap_er.private.operators.sel_lexicase_matrix.fitness_case_matrix",
        _spy_pack,
    )

    tools.sel_lexicase(population, 1, matrix=matrix, trust_matrix=True)

    assert calls == []


def test_validate_case_matrix_does_not_repack(multi_obj, make, monkeypatch):
    population = [make(multi_obj, [0], (1.0, 2.0))]
    matrix = tools.fitness_case_matrix(population)

    def _spy_pack(*args, **kwargs):
        raise AssertionError("fitness_case_matrix should not run during validate")

    monkeypatch.setattr(
        "deap_er.private.operators.sel_lexicase_matrix.fitness_case_matrix",
        _spy_pack,
    )

    tools.sel_lexicase(population, 1, matrix=matrix)

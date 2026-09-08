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

CASES_FIT = "LEX_CASES_FIT"
CASES_IND = "LEX_CASES_IND"


@pytest.fixture
def error_cases():
    creator.create_type(CASES_FIT, Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
    creator.create_type(CASES_IND, list, fitness=creator.__dict__[CASES_FIT])
    yield creator.__dict__[CASES_IND]
    del creator.__dict__[CASES_FIT]
    del creator.__dict__[CASES_IND]


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


def test_informed_cases_accepts_numpy_int_count(error_cases, make):
    values = [(0.0, 1.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0), (1.0, 1.0, 0.0, 1.0)]
    population = [make(error_cases, [i], value) for i, value in enumerate(values)]

    count: Any = numpy.int64(2)
    subset = tools.sample_informed_cases(population, count)

    assert len(subset) == 2


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

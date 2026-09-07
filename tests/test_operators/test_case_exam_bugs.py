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
import numpy
import pytest
from deap_er import Fitness, creator, tools

FIT = "EXAM_BUG_FIT"
IND = "EXAM_BUG_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(FIT, Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    yield creator.__dict__[IND]
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


@pytest.fixture
def pair_cls():
    creator.create_type("EXAM_BUG_FIT2", Fitness, weights=(-1.0, -1.0))
    creator.create_type("EXAM_BUG_IND2", list, fitness=creator.__dict__["EXAM_BUG_FIT2"])
    yield creator.__dict__["EXAM_BUG_IND2"]
    del creator.__dict__["EXAM_BUG_FIT2"]
    del creator.__dict__["EXAM_BUG_IND2"]


def _make(ind_cls, genes, values):
    individual = ind_cls(genes)
    individual.fitness.values = values
    return individual


def test_next_does_not_double_repair_after_guard(ind_cls):
    elites = [
        _make(ind_cls, [0], (0.0, 0.0, 1.0, 1.0)),
        _make(ind_cls, [1], (0.0, 1.0, 0.0, 1.0)),
    ]
    matrix = tools.fitness_case_matrix(elites)
    tools.rng.seed(4)
    guarded = tools.CaseExam.from_cases([0], 4)
    tools.guard_case_exams([guarded], elites, matrix=matrix, min_cases=2)
    tools.rng.seed(4)
    cases = tools.next_lexicase_cases(
        [tools.CaseExam.from_cases([0], 4)],
        elites,
        matrix=matrix,
        mut_prob=0.0,
        min_cases=2,
    )
    assert cases == guarded.as_cases(4)


def test_next_default_and_informed_false_keep_winner(ind_cls):
    elites = [
        _make(ind_cls, [0], (0.0, 1.0, 1.0, 0.0)),
        _make(ind_cls, [1], (1.0, 0.0, 1.0, 0.0)),
        _make(ind_cls, [2], (0.0, 0.0, 1.0, 1.0)),
    ]
    exam = tools.CaseExam.from_cases([0, 2], 4)
    expected = exam.as_cases(4)
    matrix = tools.fitness_case_matrix(elites)
    default = tools.next_lexicase_cases([exam.copy()], elites, matrix=matrix, mut_prob=0.0)
    kept = tools.next_lexicase_cases(
        [exam.copy()],
        elites,
        matrix=matrix,
        informed=False,
        mut_prob=0.0,
    )
    assert default == expected
    assert kept == expected


def test_next_informed_false_returns_mutated_mask(ind_cls):
    elites = [
        _make(ind_cls, [0], (0.0, 1.0, 1.0, 0.0)),
        _make(ind_cls, [1], (1.0, 0.0, 1.0, 0.0)),
    ]
    exam = tools.CaseExam(mask=numpy.array([True, True, False, False], dtype=bool))
    tools.rng.seed(0)
    cases = tools.next_lexicase_cases([exam], elites, informed=False, mut_prob=1.0)
    assert cases == [2, 3]


def test_score_flat_case_index_list_is_one_exam(ind_cls):
    elites = [_make(ind_cls, [0], (0.0, 1.0, 1.0, 0.0))]
    scores = tools.score_case_exams([0, 1, 2], elites)
    assert scores == [2]


def test_series_ranges_as_cases_uses_segment_indices(pair_cls):
    exam = tools.CaseExam(ranges=[(0, 256), (256, 512)])
    elites = [_make(pair_cls, [0], (1.0, 0.0))]
    assert exam.as_cases(2) == [0, 1]
    assert tools.score_case_exams([exam], elites) == [1]


def test_series_range_mutation_uses_series_length(pair_cls):
    elites = [_make(pair_cls, [0], (1.0, 0.0))]
    exam = tools.CaseExam(ranges=[(10, 20)], length=100)
    tools.rng.seed(1)
    tools.next_lexicase_cases([exam], elites, informed=False, mut_prob=1.0)
    assert exam.ranges is not None
    start, stop = exam.ranges[0]
    assert stop > 2
    assert 0 <= start <= stop <= 100


def test_guard_hamming_does_not_collapse_specialist_cases(ind_cls):
    elites = [
        _make(ind_cls, [0], (0.0, 1.0, 1.0, 1.0)),
        _make(ind_cls, [1], (0.0, 0.0, 1.0, 1.0)),
    ]
    exam = tools.CaseExam.from_cases([0, 1], 4)
    held = tools.CaseExam.from_cases([2], 4)
    tools.guard_case_exams([exam], elites, held_out=held, mode="hamming")
    assert exam.as_cases(4) == [0, 1]


def test_guard_empty_without_last_good_uses_held_out(ind_cls):
    elites = [_make(ind_cls, [0], (0.0, 1.0, 1.0, 1.0))]
    empty = tools.CaseExam.from_cases([], 4)
    held = tools.CaseExam.from_cases([3], 4)
    pool = tools.CaseExamPool([empty], held_out=held)
    tools.guard_case_exams(pool, elites)
    assert pool[0].as_cases(4) == [3]


def test_next_pool_honors_case_count_as_repair_floor(ind_cls):
    elites = [_make(ind_cls, [0], (0.0, 0.0, 0.0, 0.0))]
    tools.rng.seed(2)
    listed = tools.next_lexicase_cases(
        [tools.CaseExam.from_cases([], 4)],
        elites,
        case_count=3,
        mut_prob=0.0,
    )
    tools.rng.seed(2)
    pooled = tools.next_lexicase_cases(
        tools.CaseExamPool([tools.CaseExam.from_cases([], 4)]),
        elites,
        case_count=3,
        mut_prob=0.0,
    )
    assert listed == pooled
    assert len(pooled) >= 3

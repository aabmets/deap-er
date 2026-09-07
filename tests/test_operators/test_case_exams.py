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

FIT = "EXAM_FIT"
IND = "EXAM_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(FIT, Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    yield creator.__dict__[IND]
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def _make(ind_cls, genes, values):
    individual = ind_cls(genes)
    individual.fitness.values = values
    return individual


def test_score_unsolved_and_hamming(ind_cls):
    elites = [
        _make(ind_cls, [0], (0.0, 1.0, 0.0, 1.0)),
        _make(ind_cls, [1], (0.0, 0.0, 1.0, 1.0)),
    ]
    hard = tools.CaseExam.from_cases([1, 3], 4)
    easy = tools.CaseExam.from_cases([0], 4)

    unsolved = tools.score_case_exams([hard, easy], elites)
    hamming = tools.score_case_exams([hard, easy], elites, mode="hamming")

    assert unsolved == [1, 0]
    assert hamming == [3, 0]


def test_score_matrix_matches_pack_and_trust(ind_cls):
    elites = [_make(ind_cls, [0], (0.0, 1.0, 1.0, 0.0))]
    matrix = tools.fitness_case_matrix(elites)
    exam = tools.CaseExam.from_cases([1, 2], 4)

    default = tools.score_case_exams([exam], elites)
    packed = tools.score_case_exams([exam], elites, matrix=matrix)
    trusted = tools.score_case_exams([exam], elites, matrix=matrix, trust_matrix=True)

    assert default == packed == trusted == [2]


def test_score_custom_solved_ignores_matrix(ind_cls):
    elites = [_make(ind_cls, [0], (0.0, 1.0, 1.0, 0.0))]
    matrix = numpy.zeros((1, 4))
    exam = tools.CaseExam.from_cases([0, 1], 4)

    def solved(individual, case):
        return individual.fitness.values[case] == 0.0

    scores = tools.score_case_exams([exam], elites, matrix=matrix, solved=solved)
    assert scores == [1]


def test_guard_empty_exam_uses_last_good_then_held_out(ind_cls):
    elites = [_make(ind_cls, [0], (0.0, 1.0, 1.0, 1.0))]
    empty = tools.CaseExam.from_cases([], 4)
    held = tools.CaseExam.from_cases([2], 4)
    pool = tools.CaseExamPool([empty], held_out=held)
    pool.last_good = tools.CaseExam.from_cases([1], 4)

    tools.guard_case_exams(pool, elites)

    assert pool[0].as_cases(4) == [1]
    assert pool.last_good.as_cases(4) == [1]


def test_guard_all_solved_injects_held_out(ind_cls):
    elites = [_make(ind_cls, [0], (0.0, 0.0, 1.0, 1.0))]
    solved = tools.CaseExam.from_cases([0, 1], 4)
    held = tools.CaseExam.from_cases([2], 4)

    tools.guard_case_exams([solved], elites, held_out=held)

    assert 2 in solved.as_cases(4)


def test_guard_all_solved_bumps_via_informed(ind_cls):
    elites = [
        _make(ind_cls, [0], (0.0, 0.0, 1.0, 1.0)),
        _make(ind_cls, [1], (0.0, 1.0, 0.0, 1.0)),
    ]
    collapsed = tools.CaseExam.from_cases([0], 4)

    tools.guard_case_exams([collapsed], elites, min_cases=2)

    assert len(collapsed.as_cases(4)) >= 2


def test_next_lexicase_cases_feeds_sel_lexicase(ind_cls):
    elites = [
        _make(ind_cls, [0], (0.0, 1.0, 1.0, 0.0)),
        _make(ind_cls, [1], (1.0, 0.0, 1.0, 0.0)),
    ]
    exam = tools.CaseExam.from_cases([0, 1, 2, 3], 4)
    tools.rng.seed(11)
    cases = tools.next_lexicase_cases(
        [exam],
        elites,
        informed=False,
        mut_prob=0.0,
    )
    chosen = tools.sel_lexicase(elites, 2, cases=cases)

    assert cases
    assert all(idx in range(4) for idx in cases)
    assert all(ind in elites for ind in chosen)


def test_next_informed_matches_sample_informed_cases(ind_cls):
    elites = [
        _make(ind_cls, [0], (0.0, 1.0, 1.0, 0.0)),
        _make(ind_cls, [1], (1.0, 0.0, 1.0, 0.0)),
        _make(ind_cls, [2], (0.0, 0.0, 1.0, 1.0)),
    ]
    exam = tools.CaseExam.from_cases([0, 2], 4)
    matrix = tools.fitness_case_matrix(elites)

    tools.rng.seed(5)
    informed = tools.sample_informed_cases(elites, 2, matrix=matrix)
    tools.rng.seed(5)
    cases = tools.next_lexicase_cases(
        [exam],
        elites,
        matrix=matrix,
        case_count=2,
        informed=True,
        mut_prob=0.0,
    )

    assert cases == informed


def test_next_empty_exams_or_elites_raises(ind_cls):
    elites = [_make(ind_cls, [0], (0.0, 1.0, 1.0, 1.0))]
    with pytest.raises(ValueError, match="non-empty"):
        tools.next_lexicase_cases([], elites, mut_prob=0.0)
    with pytest.raises(ValueError, match="non-empty"):
        tools.score_case_exams([tools.CaseExam.from_cases([0], 4)], [])


def test_score_unknown_mode_raises(ind_cls):
    elites = [_make(ind_cls, [0], (0.0, 1.0, 1.0, 1.0))]
    exam = tools.CaseExam.from_cases([1], 4)
    with pytest.raises(ValueError, match="unsolved"):
        tools.score_case_exams([exam], elites, mode="sharpe")

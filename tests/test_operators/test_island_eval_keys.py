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
from deap_er import Fitness, Toolbox, creator, tools

ISL_EVAL_FIT = "ISL_EVAL_FIT"
ISL_EVAL_IND = "ISL_EVAL_IND"


def test_island_eval_keys_mask_and_ranges_same_cases_match():
    mask_exam = tools.CaseExam.from_cases([0, 2], 4)
    ranges_exam = tools.CaseExam(ranges=[(0, 1), (2, 3)])
    keys = tools.island_eval_keys([mask_exam, ranges_exam], n_cases=4)

    assert keys[0] == keys[1]


def test_island_eval_keys_same_exam_matches():
    exam = tools.CaseExam.from_cases([0, 2], 4)
    keys = tools.island_eval_keys([exam, exam.copy()], n_cases=4)

    assert keys[0] == keys[1]


def test_island_eval_keys_different_exams_differ():
    left = tools.CaseExam.from_cases([0, 1], 4)
    right = tools.CaseExam.from_cases([2, 3], 4)
    keys = tools.island_eval_keys([left, right], n_cases=4)

    assert keys[0] != keys[1]


def test_island_eval_keys_shared_matrix_matches():
    exam = tools.CaseExam.from_cases([1], 4)
    matrix = numpy.zeros((8, 2))
    keys = tools.island_eval_keys(
        [exam, exam.copy()],
        n_cases=4,
        matrix=matrix,
    )

    assert keys[0] == keys[1]


def test_island_eval_keys_different_matrices_differ():
    exam = tools.CaseExam.from_cases([1], 4)
    keys = tools.island_eval_keys(
        [exam, exam.copy()],
        n_cases=4,
        matrices=[numpy.zeros((8, 2)), numpy.zeros((9, 2))],
    )

    assert keys[0] != keys[1]


def test_island_eval_keys_rejects_matrix_and_matrices():
    exam = tools.CaseExam.from_cases([0], 4)
    matrix = numpy.zeros((4, 1))

    with pytest.raises(ValueError, match="matrix or matrices"):
        tools.island_eval_keys([exam], n_cases=4, matrix=matrix, matrices=[matrix])


@pytest.fixture
def ind_cls():
    creator.create_type(ISL_EVAL_FIT, Fitness, weights=(1.0,))
    creator.create_type(ISL_EVAL_IND, list, fitness=creator.__dict__[ISL_EVAL_FIT])
    yield creator.__dict__[ISL_EVAL_IND]
    del creator.__dict__[ISL_EVAL_FIT]
    del creator.__dict__[ISL_EVAL_IND]


def test_step_islands_accepts_mig_fully_connected(ind_cls):
    def migrate(populations):
        tools.mig_fully_connected(populations, 1, tools.sel_best)

    first = [ind_cls([1]), ind_cls([2])]
    second = [ind_cls([8]), ind_cls([9])]
    for ind in first + second:
        ind.fitness.values = (float(ind[0]),)
    keep = Toolbox()
    keep.register("vary", lambda pop: list(pop))
    keep.register("select", tools.sel_best)
    keep.register("evaluate", lambda ind: (float(ind[0]),))

    tools.step_islands([(keep, first), (keep, second)], migrate=migrate)

    assert all(len(deme) == 2 for deme in (first, second))


def test_step_islands_invalidates_with_island_eval_keys(ind_cls):
    shared = tools.CaseExam.from_cases([0, 1], 4)
    distinct = tools.CaseExam.from_cases([2, 3], 4)
    keys = tools.island_eval_keys([shared, distinct], n_cases=4)

    first = [ind_cls([1]), ind_cls([2])]
    second = [ind_cls([8]), ind_cls([9])]
    for ind in first + second:
        ind.fitness.values = (float(ind[0]),)
    first_ids = {id(ind) for ind in first}
    keep = Toolbox()
    keep.register("vary", lambda pop: list(pop))
    keep.register("select", tools.sel_best)
    keep.register("evaluate", lambda ind: (float(ind[0]),))

    tools.step_islands(
        [(keep, first), (keep, second)],
        migrate=lambda pops: tools.mig_ring(pops, 1, tools.sel_best),
        eval_keys=keys,
    )

    arrivals = [ind for ind in first if id(ind) not in first_ids]
    assert arrivals
    assert all(not ind.fitness.is_valid() for ind in arrivals)

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
from deap_er import tools


def test_case_exam_converts_mask_and_ranges():
    mask = numpy.array([False, True, True, False, True], dtype=bool)
    exam = tools.CaseExam(mask=mask)

    assert exam.as_ranges(5) == [(1, 3), (4, 5)]
    assert numpy.array_equal(exam.as_mask(5), mask)
    assert exam.as_cases(5) == [1, 2, 4]


def test_case_exam_ranges_paint_mask_and_catalog_indices():
    exam = tools.CaseExam(ranges=[(0, 2), (3, 4)])

    assert exam.as_ranges(5) == [(0, 2), (3, 4)]
    assert numpy.array_equal(exam.as_mask(5), [True, True, False, True, False])
    assert exam.as_cases(5) == [0, 1, 3]


def test_case_exam_from_cases_and_copy_assign():
    exam = tools.CaseExam.from_cases([0, 2], 4)
    clone = exam.copy()
    assert clone.mask is not None
    clone.mask[1] = True
    other = tools.CaseExam(ranges=[(1, 3)])
    exam.assign(other)

    assert exam.as_cases(4) == [1, 2]
    assert clone.as_cases(4) == [0, 1, 2]


def test_case_exam_rejects_both_neither_and_bad_mask():
    with pytest.raises(ValueError, match="exactly one"):
        tools.CaseExam()
    with pytest.raises(ValueError, match="exactly one"):
        tools.CaseExam(ranges=[(0, 1)], mask=numpy.array([True]))
    with pytest.raises(ValueError, match="one-dimensional"):
        tools.CaseExam(mask=numpy.array([[True]]))
    with pytest.raises(ValueError, match="dtype bool"):
        tools.CaseExam(mask=numpy.array([1, 0]))


def test_case_exam_from_cases_rejects_bad_index():
    with pytest.raises(ValueError, match="non-negative"):
        tools.CaseExam.from_cases([0], -1)
    with pytest.raises(IndexError, match="case index"):
        tools.CaseExam.from_cases([4], 3)
    with pytest.raises(IndexError, match="case index"):
        tools.CaseExam.from_cases([True], 2)


def test_case_exam_pool_stores_held_out_and_min_cases():
    first = tools.CaseExam.from_cases([0], 3)
    held = tools.CaseExam.from_cases([2], 3)
    pool = tools.CaseExamPool([first], held_out=held, min_cases=2)

    assert len(pool) == 1
    assert pool[0] is first
    assert list(pool) == [first]
    assert pool.held_out is held
    assert pool.min_cases == 2
    assert pool.last_good is None
    with pytest.raises(ValueError, match="at least 1"):
        tools.CaseExamPool([], min_cases=0)


def test_coerce_case_exam_accepts_mask_ranges_and_indices():
    mask = numpy.array([True, False, True], dtype=bool)
    assert tools.coerce_case_exam(mask).as_cases(3) == [0, 2]
    assert tools.coerce_case_exam([(0, 2)]).as_ranges(4) == [(0, 2)]
    assert tools.coerce_case_exam([1, 3], n_cases=4).as_cases(4) == [1, 3]
    empty = tools.coerce_case_exam([])
    assert empty.as_cases(3) == []
    with pytest.raises(ValueError, match="n_cases"):
        tools.coerce_case_exam([0, 1])

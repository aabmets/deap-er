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

NAN = numpy.nan

FIT = "CASE_FIT"
IND = "CASE_IND"


def _setup() -> None:
    creator.create_type(FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])


def _teardown() -> None:
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def test_case_errors_returns_mse_for_one_range():
    predicted = numpy.array([1.0, 2.0, 3.0])
    target = numpy.zeros(3)
    errors = tools.case_errors(predicted, target, [(0, 3)])
    assert errors == (pytest.approx(14.0 / 3.0),)


def test_case_errors_scores_disjoint_ranges_independently():
    predicted = numpy.array([0.0, 1.0, 2.0, 3.0])
    target = numpy.array([0.0, 0.0, 1.0, 1.0])
    errors = tools.case_errors(predicted, target, [(0, 2), (2, 4)])
    assert errors == (pytest.approx(0.5), pytest.approx(2.5))


def test_case_errors_ignores_non_finite_warmup():
    predicted = numpy.array([NAN, NAN, 1.0, 3.0])
    target = numpy.array([NAN, NAN, 1.0, 1.0])
    errors = tools.case_errors(predicted, target, [(0, 4)])
    assert errors == (pytest.approx(2.0),)


def test_case_errors_splits_contiguous_true_runs_from_a_mask():
    predicted = numpy.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    target = numpy.zeros(7)
    mask = numpy.array([False, True, True, False, True, True, False])
    errors = tools.case_errors(predicted, target, mask)
    assert errors == (pytest.approx(2.5), pytest.approx(20.5))


def test_case_errors_honors_an_explicit_valid_mask():
    predicted = numpy.array([9.0, 1.0, 2.0])
    target = numpy.array([0.0, 0.0, 0.0])
    valid = numpy.array([False, True, True])
    errors = tools.case_errors(predicted, target, [(0, 3)], valid=valid)
    assert errors == (pytest.approx(2.5),)


def test_case_errors_returns_empty_for_no_cases():
    predicted = numpy.array([1.0, 2.0])
    target = numpy.array([0.0, 0.0])
    assert tools.case_errors(predicted, target, []) == ()
    assert tools.case_errors(predicted, target, numpy.zeros(2, dtype=bool)) == ()


def test_case_errors_uses_empty_when_a_case_has_no_scorable_samples():
    predicted = numpy.array([NAN, NAN])
    target = numpy.array([NAN, NAN])
    assert tools.case_errors(predicted, target, [(0, 2)]) == (float("inf"),)
    assert tools.case_errors(predicted, target, [(0, 2)], empty=1.0e6) == (1.0e6,)


def test_case_errors_returns_empty_when_an_interval_is_empty():
    predicted = numpy.array([1.0, 2.0, 3.0])
    target = numpy.zeros(3)
    errors = tools.case_errors(predicted, target, [(1, 1), (0, 3)])
    assert errors == (float("inf"), pytest.approx(14.0 / 3.0))


def test_case_errors_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        tools.case_errors(numpy.array([1.0, 2.0]), numpy.array([1.0]), [(0, 1)])


def test_case_errors_rejects_non_one_dimensional_input():
    with pytest.raises(ValueError, match="one-dimensional"):
        tools.case_errors(numpy.array([[1.0]]), numpy.array([1.0]), [(0, 1)])


def test_case_errors_rejects_invalid_range_endpoints():
    predicted = numpy.array([1.0, 2.0, 3.0])
    target = numpy.zeros(3)
    with pytest.raises(ValueError, match="range endpoints"):
        tools.case_errors(predicted, target, [(2, 1)])
    with pytest.raises(ValueError, match="range endpoints"):
        tools.case_errors(predicted, target, [(0, 4)])
    with pytest.raises(ValueError, match="range endpoints"):
        tools.case_errors(predicted, target, [(-1, 2)])


def test_case_errors_rejects_invalid_masks():
    predicted = numpy.array([1.0, 2.0])
    target = numpy.zeros(2)
    with pytest.raises(ValueError, match="dtype bool"):
        tools.case_errors(predicted, target, numpy.array([1, 0], dtype=int))
    with pytest.raises(ValueError, match="match the series length"):
        tools.case_errors(predicted, target, numpy.array([True, False, True]))


def test_case_errors_fits_lexicase_fitness_vectors():
    _setup()
    try:
        better = creator.__dict__[IND]([])
        worse = creator.__dict__[IND]([])
        better.fitness.values = (0.1, 0.2)
        worse.fitness.values = (0.9, 0.8)
        chosen = tools.sel_lexicase([worse, better], 1, cases=[0, 1])
    finally:
        _teardown()
    assert chosen == [better]

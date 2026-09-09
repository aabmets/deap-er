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

NAN = numpy.nan


def test_affine_case_errors_matches_affine_scale_then_case_errors():
    predicted = numpy.array([0.0, 1.0, 2.0, 3.0])
    target = numpy.array([1.0, 3.0, 5.0, 7.0])
    cases = [(0, 4)]
    intercept, slope = tools.affine_scale(predicted, target)
    scaled = intercept + slope * predicted
    expected = tools.case_errors(scaled, target, cases)
    fitted = tools.affine_case_errors(predicted, target, cases)
    numpy.testing.assert_allclose(fitted, expected)
    assert fitted[0] < tools.case_errors(predicted, target, cases)[0]


def test_affine_case_errors_honors_valid_mask():
    predicted = numpy.array([0.0, 1.0, 2.0, 3.0])
    target = numpy.array([99.0, 3.0, 5.0, 7.0])
    valid = numpy.array([False, True, True, True])
    cases = [(0, 4)]
    intercept, slope = tools.affine_scale(predicted, target, valid=valid)
    scaled = intercept + slope * predicted
    expected = tools.case_errors(scaled, target, cases, valid=valid)
    fitted = tools.affine_case_errors(predicted, target, cases, valid=valid)
    numpy.testing.assert_allclose(fitted, expected)


def test_affine_case_errors_identity_when_no_scorable_samples():
    predicted = numpy.array([NAN, NAN])
    target = numpy.array([NAN, NAN])
    cases = [(0, 2)]
    assert tools.affine_case_errors(predicted, target, cases) == tools.case_errors(
        predicted,
        target,
        cases,
    )


def test_affine_case_errors_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="one-dimensional"):
        tools.affine_case_errors(numpy.array([[1.0, 2.0]]), numpy.array([1.0, 2.0]), [(0, 2)])

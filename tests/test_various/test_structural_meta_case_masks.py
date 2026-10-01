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
import operator

import numpy
import pytest
from deap_er import gp, tools

NAN = numpy.nan


def _population():
    pset = gp.PrimitiveSetTyped("main", [float, float], float)
    pset.add_primitive(operator.add, [float, float], float, name="vadd")
    tree = gp.PrimitiveTree.from_string("vadd(ARG0, ARG1)", pset)
    return [tree, tree]


def _blank_share(predicted, valid):
    matrix = tools.structural_meta_case_columns(
        _population(), predicted=predicted, columns=("non_finite_fraction",), valid=valid
    )
    return matrix[:, 0]


def test_per_individual_mask_excludes_each_program_lookback():
    predicted = numpy.array(
        [
            [NAN, 1.0, 2.0, 3.0],
            [NAN, NAN, NAN, 3.0],
        ]
    )
    valid = numpy.array(
        [
            [False, True, True, True],
            [False, False, False, True],
        ]
    )
    assert _blank_share(predicted, valid) == pytest.approx([0.0, 0.0])


def test_per_individual_mask_matches_shared_mask_when_rows_are_equal():
    predicted = numpy.array([[NAN, 1.0, NAN], [1.0, NAN, 2.0]])
    shared = numpy.array([False, True, True])
    tiled = numpy.tile(shared, (2, 1))
    assert _blank_share(predicted, tiled) == pytest.approx(_blank_share(predicted, shared))


def test_per_individual_mask_with_no_true_row_is_nan_for_that_individual():
    predicted = numpy.array([[1.0, NAN], [1.0, 2.0]])
    valid = numpy.array([[False, False], [True, True]])
    shares = _blank_share(predicted, valid)
    assert numpy.isnan(shares[0])
    assert shares[1] == pytest.approx(0.0)


def test_per_individual_mask_of_the_wrong_shape_raises():
    predicted = numpy.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match=r"shape \(2, 3\)"):
        _blank_share(predicted, numpy.ones((3, 3), dtype=bool))

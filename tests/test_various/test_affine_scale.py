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
from deap_er import Fitness, creator, gp, tools

NAN = numpy.nan
FIT = "AFFINE_FIT"
IND = "AFFINE_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, gp.PrimitiveTree, fitness=creator.__dict__[FIT])
    yield creator.__dict__[IND]
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def test_affine_scale_fits_keijzer_line():
    predicted = numpy.array([1.0, 2.0, 3.0, 4.0])
    target = 1.0 + 2.0 * predicted
    intercept, slope = tools.affine_scale(predicted, target)
    numpy.testing.assert_allclose([intercept, slope], [1.0, 2.0])


def test_affine_scale_honors_case_errors_valid_mask():
    predicted = numpy.array([0.0, 1.0, 2.0, 3.0])
    target = numpy.array([99.0, 3.0, 5.0, 7.0])
    valid = numpy.array([False, True, True, True])
    intercept, slope = tools.affine_scale(predicted, target, valid=valid)
    numpy.testing.assert_allclose([intercept, slope], [1.0, 2.0])


def test_affine_scale_skips_non_finite_samples():
    predicted = numpy.array([NAN, 1.0, 2.0, 3.0])
    target = numpy.array([0.0, 3.0, 5.0, 7.0])
    intercept, slope = tools.affine_scale(predicted, target)
    numpy.testing.assert_allclose([intercept, slope], [1.0, 2.0])


def test_affine_scale_identity_when_no_scorable_samples():
    predicted = numpy.array([NAN, NAN])
    target = numpy.array([NAN, NAN])
    assert tools.affine_scale(predicted, target) == (0.0, 1.0)
    assert tools.affine_scale(predicted, target, valid=numpy.zeros(2, dtype=bool)) == (0.0, 1.0)


def test_affine_scale_constant_prediction_is_intercept_only():
    predicted = numpy.array([2.0, 2.0, 2.0])
    target = numpy.array([5.0, 7.0, 9.0])
    intercept, slope = tools.affine_scale(predicted, target)
    numpy.testing.assert_allclose([intercept, slope], [5.0, 1.0])


def test_affine_scale_darwinian_lowers_case_errors_without_a_tree():
    predicted = numpy.array([0.0, 1.0, 2.0, 3.0])
    target = numpy.array([1.0, 3.0, 5.0, 7.0])
    intercept, slope = tools.affine_scale(predicted, target)
    scaled = intercept + slope * predicted
    raw = tools.case_errors(predicted, target, [(0, 4)])
    fitted = tools.case_errors(scaled, target, [(0, 4)])
    assert fitted[0] < raw[0]
    numpy.testing.assert_allclose(fitted[0], 0.0)


def test_affine_scale_darwinian_leaves_tree_and_fitness_untouched(ind_cls):
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_ephemeral_constant("AFFINE_DARWIN", lambda: 0.25)
    eph = pset.terminals[object][-1]
    tree = ind_cls([pset.mapping["add"], eph(), pset.mapping["ARG0"]])
    gp.assign_ephemerals(tree, [0.5])
    tree.fitness.values = (1.25,)
    before = list(tree)
    compiled = gp.compile_tree(tree, pset)
    predicted = numpy.array([compiled(x) for x in (0.0, 1.0, 2.0)])
    target = numpy.array([1.0, 3.0, 5.0])

    intercept, slope = gp.affine_scale(predicted, target)

    assert tree.fitness.values == (1.25,)
    assert list(tree) == before
    assert gp.compile_tree(tree, pset) is compiled
    numpy.testing.assert_allclose(intercept + slope * predicted, target)


def test_affine_scale_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="one-dimensional"):
        tools.affine_scale([[1.0, 2.0]], [1.0, 2.0])
    with pytest.raises(ValueError, match="same length"):
        tools.affine_scale([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="valid"):
        tools.affine_scale([1.0, 2.0], [1.0, 2.0], valid=[True])

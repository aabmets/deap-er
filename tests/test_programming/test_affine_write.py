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

FIT = "AFFINE_WRITE_FIT"
IND = "AFFINE_WRITE_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, gp.PrimitiveTree, fitness=creator.__dict__[FIT])
    yield creator.__dict__[IND]
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def _float_pset(name: str) -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.mul, 2)
    pset.add_ephemeral_constant(name, lambda: 0.25)
    return pset


def _arg_tree(ind_cls, pset: gp.PrimitiveSet):
    return ind_cls([pset.mapping["ARG0"]])


def test_write_affine_scale_wraps_tree_and_invalidates_fitness(ind_cls):
    pset = _float_pset("AFFINE_WRITE_TREE")
    tree = _arg_tree(ind_cls, pset)
    tree.fitness.values = (4.0,)
    predicted = numpy.array([0.0, 1.0, 2.0, 3.0])
    target = 1.0 + 2.0 * predicted
    intercept, slope = tools.affine_scale(predicted, target)

    written = gp.write_affine_scale(tree, intercept, slope, pset)
    func = gp.compile_tree(tree, pset)

    assert written is tree
    assert not tree.fitness.is_valid()
    numpy.testing.assert_allclose([func(x) for x in predicted], target)
    numpy.testing.assert_allclose(gp.extract_ephemerals(tree), [intercept, slope])


def test_write_affine_scale_invalidates_compile_cache(ind_cls):
    pset = _float_pset("AFFINE_WRITE_CACHE")
    tree = _arg_tree(ind_cls, pset)
    tree.fitness.values = (1.0,)
    old = gp.PrimitiveTree(list(tree))
    compiled = gp.compile_tree(tree, pset)

    gp.write_affine_scale(tree, 1.0, 3.0, pset)

    assert gp.compile_tree(old, pset) is not compiled
    assert gp.compile_tree(tree, pset)(1.0) == pytest.approx(4.0)


def test_write_affine_scale_slim_wrapping_delta_and_cache(ind_cls):
    pset = _float_pset("AFFINE_WRITE_SLIM")
    head = _arg_tree(ind_cls, pset)
    delta = ind_cls([pset.terminals[object][-1]()])
    gp.assign_ephemerals(delta, [1.0])
    slim = gp.SlimTree(head, [delta])
    slim.fitness = type(head.fitness)()
    slim.fitness.values = (2.0,)
    old_head = gp.PrimitiveTree(list(slim.head))
    old_delta = gp.PrimitiveTree(list(slim.deltas[0]))
    compiled_head = gp.compile_tree(slim.head, pset)
    compiled_delta = gp.compile_tree(slim.deltas[0], pset)
    xs = numpy.array([0.0, 1.0, 2.0])
    predicted = numpy.array([gp.compile_slim_tree(slim, pset)(x) for x in xs])
    target = 2.0 + 3.0 * predicted
    intercept, slope = tools.affine_scale(predicted, target)

    written = gp.write_affine_scale(slim, intercept, slope, pset)
    scaled = numpy.array([gp.compile_slim_tree(slim, pset)(x) for x in xs])

    assert written is slim
    assert not slim.fitness.is_valid()
    numpy.testing.assert_allclose(scaled, target)
    assert gp.compile_tree(old_head, pset) is not compiled_head
    assert gp.compile_tree(old_delta, pset) is not compiled_delta


def test_write_affine_scale_requires_add_and_mul(ind_cls):
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    tree = _arg_tree(ind_cls, pset)
    with pytest.raises(TypeError, match="multiplication"):
        gp.write_affine_scale(tree, 0.0, 1.0, pset)

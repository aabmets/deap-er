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
import random

import pytest
from deap_er import base, creator
from deap_er.gp.generators import gen_full
from deap_er.gp.mutation import mut_shrink
from deap_er.gp.primitives import PrimitiveSet, PrimitiveSetTyped, PrimitiveTree

MUT_FIT = "MUT_FIT"
MUT_IND = "MUT_IND"


@pytest.fixture
def ind_cls():
    creator.create(MUT_FIT, base.Fitness, weights=(-1.0,))
    creator.create(MUT_IND, PrimitiveTree, fitness=creator.__dict__[MUT_FIT])
    yield creator.__dict__[MUT_IND]
    del creator.__dict__[MUT_FIT]
    del creator.__dict__[MUT_IND]


def _untyped_pset() -> PrimitiveSet:
    pset = PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.mul, 2)
    return pset


@pytest.mark.parametrize("seed", range(10))
def test_mut_shrink_shrinks_untyped_trees(ind_cls, seed):
    # Every primitive in an untyped set has args equal to its return type,
    # so the operator has to accept those arguments as replacements.
    pset = _untyped_pset()
    random.seed(seed)
    tree = ind_cls(gen_full(pset, min_depth=2, max_depth=3))
    before = len(tree)

    (mutant,) = mut_shrink(tree)

    assert len(mutant) < before


def test_mut_shrink_keeps_tree_printable(ind_cls):
    pset = _untyped_pset()
    random.seed(11)
    tree = ind_cls(gen_full(pset, min_depth=2, max_depth=3))

    (mutant,) = mut_shrink(tree)

    assert str(mutant)
    assert all(node.arity <= 2 for node in mutant if hasattr(node, "arity"))


def test_mut_shrink_leaves_tiny_trees_alone(ind_cls):
    pset = _untyped_pset()
    tree = ind_cls(PrimitiveTree.from_string("ARG0", pset))

    (mutant,) = mut_shrink(tree)

    assert len(mutant) == 1


def test_mut_shrink_substitutes_argument_of_matching_type(ind_cls):
    pset = PrimitiveSetTyped("main", [float], float)
    pset.add_primitive(operator.add, [float, float], float)
    random.seed(5)
    tree = ind_cls(PrimitiveTree.from_string("add(ARG0, add(ARG0, ARG0))", pset))

    (mutant,) = mut_shrink(tree)

    assert len(mutant) < 5

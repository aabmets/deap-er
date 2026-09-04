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

import pytest
from deap_er import base, creator, tools
from deap_er.gp.generators import gen_full
from deap_er.gp.mutation import (
    mut_ephemeral,
    mut_insert,
    mut_node_replacement,
    mut_shrink,
    mut_uniform,
)
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
    tools.seed(seed)
    tree = ind_cls(gen_full(pset, min_depth=2, max_depth=3))
    before = len(tree)

    (mutant,) = mut_shrink(tree)

    assert len(mutant) < before


def test_mut_shrink_keeps_tree_printable(ind_cls):
    pset = _untyped_pset()
    tools.seed(11)
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
    tools.seed(5)
    tree = ind_cls(PrimitiveTree.from_string("add(ARG0, add(ARG0, ARG0))", pset))

    (mutant,) = mut_shrink(tree)

    assert len(mutant) < 5


def test_mut_uniform_replaces_a_subtree(ind_cls):
    pset = _untyped_pset()
    tools.seed(21)
    tree = ind_cls(gen_full(pset, min_depth=2, max_depth=2))
    before = str(tree)

    (mutant,) = mut_uniform(tree, lambda prim_set, ret_type: gen_full(prim_set, 1, 1), pset)

    assert str(mutant)
    assert str(mutant) != before or len(mutant) >= 1


def test_mut_node_replacement_leaves_tiny_trees_alone(ind_cls):
    pset = _untyped_pset()
    tree = ind_cls(PrimitiveTree.from_string("ARG0", pset))

    (mutant,) = mut_node_replacement(tree, pset)

    assert len(mutant) == 1


def test_mut_node_replacement_swaps_a_compatible_node(ind_cls):
    pset = _untyped_pset()
    tools.seed(22)
    tree = ind_cls(gen_full(pset, min_depth=2, max_depth=3))

    (mutant,) = mut_node_replacement(tree, pset)

    assert str(mutant)


def test_mut_insert_grows_or_keeps_the_tree(ind_cls):
    pset = _untyped_pset()
    tools.seed(23)
    tree = ind_cls(gen_full(pset, min_depth=1, max_depth=2))
    before = len(tree)

    (mutant,) = mut_insert(tree, pset)

    assert len(mutant) >= before


def test_mut_insert_returns_unchanged_when_no_compatible_primitive(ind_cls):
    pset = PrimitiveSetTyped("main", [float], float)
    pset.add_primitive(operator.add, [int, int], float)
    tree = ind_cls(PrimitiveTree.from_string("ARG0", pset))

    (mutant,) = mut_insert(tree, pset)

    assert list(mutant) == list(tree)


def test_mut_ephemeral_modes(ind_cls):
    pset = PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_ephemeral_constant("COV_EPH", lambda: 1)
    tools.seed(24)
    tree = ind_cls(gen_full(pset, min_depth=1, max_depth=2))

    (one,) = mut_ephemeral(tree, mode="one")
    (all_,) = mut_ephemeral(one, mode="all")
    empty = ind_cls(PrimitiveTree.from_string("ARG0", pset))
    (unchanged,) = mut_ephemeral(empty, mode="all")

    assert str(all_)
    assert len(unchanged) == 1
    with pytest.raises(ValueError, match="Mode must be"):
        mut_ephemeral(tree, mode="neither")

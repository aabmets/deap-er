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
from deap_er import gp


def _add_pset() -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    return pset


def test_from_string_round_trip_named_and_literal():
    pset = _add_pset()
    tree = gp.PrimitiveTree.from_string("add(ARG0, 2)", pset)
    restored = gp.PrimitiveTree.from_string(str(tree), pset)

    assert [node.name for node in restored] == [node.name for node in tree]
    assert restored[-1].value == 2
    assert gp.compile_tree(restored, pset)(3) == 5


def test_from_string_literal_kinds():
    pset = _add_pset()
    pset.add_primitive(operator.mul, 2)
    tree = gp.PrimitiveTree.from_string("add(mul(ARG0, 1.5), True)", pset)
    values = [node.value for node in tree if not hasattr(node, "args")]

    assert 1.5 in values
    assert True in values
    assert gp.compile_tree(tree, pset)(2) == 4.0


def test_from_string_rejects_non_literal():
    pset = _add_pset()
    with pytest.raises(TypeError, match="Unable to evaluate terminal"):
        gp.PrimitiveTree.from_string("add(ARG0, foo)", pset)


def test_from_string_rejects_an_extra_argument():
    pset = _add_pset()
    with pytest.raises(TypeError, match="extra"):
        gp.PrimitiveTree.from_string("add(ARG0, 2, 3)", pset)


def test_from_string_rejects_a_trailing_literal():
    pset = _add_pset()
    with pytest.raises(TypeError, match="extra"):
        gp.PrimitiveTree.from_string("add(ARG0, 2) 99", pset)


def test_from_string_rejects_an_incomplete_call():
    pset = _add_pset()
    with pytest.raises(TypeError, match="incomplete"):
        gp.PrimitiveTree.from_string("add(ARG0)", pset)


def test_setitem_rejects_arity_changes():
    pset = _add_pset()
    tree = gp.PrimitiveTree.from_string("add(ARG0, 2)", pset)
    with pytest.raises(ValueError, match="different arity"):
        tree[0] = tree[-1]
    with pytest.raises(IndexError, match="slice larger"):
        tree[10:] = tree
    with pytest.raises(ValueError, match="subtree with an arity"):
        tree[0:] = [tree[-1], tree[-1]]


def test_setitem_open_start_slice_replaces_whole_tree():
    pset = _add_pset()
    tree = gp.PrimitiveTree.from_string("add(ARG0, 2)", pset)
    replacement = gp.PrimitiveTree.from_string("ARG0", pset)

    tree[:] = replacement

    assert str(tree) == "ARG0"
    assert len(tree) == 1
    assert gp.compile_tree(tree, pset)(7) == 7


def test_setitem_stop_only_slice_replaces_whole_tree():
    pset = _add_pset()
    tree = gp.PrimitiveTree.from_string("add(ARG0, 2)", pset)
    replacement = gp.PrimitiveTree.from_string("ARG0", pset)

    tree[: len(tree)] = replacement

    assert str(tree) == "ARG0"
    assert list(tree) == list(replacement)

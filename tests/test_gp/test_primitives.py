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
from typing import Any

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


def test_add_primitive_rejects_duplicate_inferred_name():
    pset = _add_pset()
    with pytest.raises(ValueError, match="unique name"):
        pset.add_primitive(operator.add, 2)


def test_add_primitive_rejects_duplicate_explicit_name():
    pset = _add_pset()
    with pytest.raises(ValueError, match="unique name"):
        pset.add_primitive(operator.mul, 2, name="add")


def test_add_primitive_allows_renamed_duplicate():
    pset = _add_pset()
    pset.add_primitive(operator.add, 2, name="add2")

    assert "add2" in pset.context
    assert pset.prims_count == 2


def test_add_terminal_rejects_duplicate_name():
    pset = _add_pset()
    pset.add_terminal(1.0, name="one")
    with pytest.raises(ValueError, match="unique name"):
        pset.add_terminal(2.0, name="one")


def test_add_terminal_allows_unnamed_values():
    pset = _add_pset()
    before = pset.terms_count  # the ARG0 terminal is registered by the constructor
    pset.add_terminal(1.0)
    pset.add_terminal(2.0)

    assert pset.terms_count == before + 2


def _zero() -> int:
    return 0


def test_untyped_set_rejects_zero_arity_and_adds_ephemeral():
    pset = gp.PrimitiveSet("main", 1)
    with pytest.raises(ValueError, match="arity should be"):
        pset.add_primitive(operator.add, 0)
    pset.add_ephemeral_constant("COV_EPH_PRIM", _zero)
    pset.add_ephemeral_constant("COV_EPH_PRIM", _zero)
    assert pset.terms_count >= 2
    with pytest.raises(TypeError, match="named differently even between psets"):
        pset.add_ephemeral_constant("COV_EPH_PRIM", lambda: 1)
    with pytest.raises(TypeError, match="gp module"):
        pset.add_ephemeral_constant("Terminal", _zero)


def test_rename_arguments_and_adf():
    adf = gp.PrimitiveSet("ADF0", 1)
    adf.add_primitive(operator.add, 2)
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.add_adf(adf)
    pset.rename_arguments(ARG0="x")
    assert pset.arguments == ["x"]
    assert "ADF0" in pset.mapping


def test_primitive_and_terminal_equality():
    pset = _add_pset()
    first = pset.mapping["add"]
    second = gp.PrimitiveSet("other", 1)
    second.add_primitive(operator.add, 2)
    assert first == second.mapping["add"]
    assert first != object()
    term = pset.mapping["ARG0"]
    assert term == pset.mapping["ARG0"]
    assert term != object()


def test_setitem_rejects_arity_changes():
    pset = _add_pset()
    tree = gp.PrimitiveTree.from_string("add(ARG0, 2)", pset)
    with pytest.raises(ValueError, match="different arity"):
        tree[0] = tree[-1]
    with pytest.raises(IndexError, match="slice larger"):
        tree[10:] = tree
    with pytest.raises(ValueError, match="subtree with an arity"):
        tree[0:] = [tree[-1], tree[-1]]


def test_from_string_rejects_type_mismatches():
    pset = gp.PrimitiveSetTyped("main", [float], float)
    pset.add_primitive(operator.add, [float, float], float)
    pset.add_primitive(operator.neg, [int], int, name="neg")
    with pytest.raises(TypeError, match="does not match"):
        gp.PrimitiveTree.from_string("add(ARG0, True)", pset)
    with pytest.raises(TypeError, match="return type"):
        gp.PrimitiveTree.from_string("add(neg(1), ARG0)", pset)


def test_add_primitive_requires_a_name():
    pset = gp.PrimitiveSetTyped("main", [float], float)
    nameless: Any = object()
    with pytest.raises(TypeError, match="__name__"):
        pset.add_primitive(nameless, [float], float)


def test_add_primitive_rejects_non_positive_weight():
    pset = gp.PrimitiveSet("main", 1)
    with pytest.raises(ValueError, match="weight"):
        pset.add_primitive(operator.add, 2, weight=0.0)

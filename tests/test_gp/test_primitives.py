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
from deap_er.gp.primitives import PrimitiveSet, PrimitiveTree
from deap_er.gp.tools import compile_tree


def _add_pset() -> PrimitiveSet:
    pset = PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    return pset


def test_from_string_round_trip_named_and_literal():
    pset = _add_pset()
    tree = PrimitiveTree.from_string("add(ARG0, 2)", pset)
    restored = PrimitiveTree.from_string(str(tree), pset)

    assert [node.name for node in restored] == [node.name for node in tree]
    assert restored[-1].value == 2
    assert compile_tree(restored, pset)(3) == 5


def test_from_string_literal_kinds():
    pset = _add_pset()
    pset.add_primitive(operator.mul, 2)
    tree = PrimitiveTree.from_string("add(mul(ARG0, 1.5), True)", pset)
    values = [node.value for node in tree if not hasattr(node, "args")]

    assert 1.5 in values
    assert True in values
    assert compile_tree(tree, pset)(2) == 4.0


def test_from_string_rejects_non_literal():
    pset = _add_pset()
    with pytest.raises(TypeError, match="Unable to evaluate terminal"):
        PrimitiveTree.from_string("add(ARG0, foo)", pset)


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

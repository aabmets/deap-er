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


def test_add_primitive_rejects_non_positive_weight():
    pset = gp.PrimitiveSet("main", 1)
    with pytest.raises(ValueError, match="weight"):
        pset.add_primitive(operator.add, 2, weight=0.0)

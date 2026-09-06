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


def test_add_primitive_rejects_name_that_matches_an_argument():
    pset = gp.PrimitiveSetTyped("main", [float], float)
    pset.rename_arguments(ARG0="price")
    with pytest.raises(ValueError, match="shadow"):
        pset.add_primitive(operator.neg, [float], float, name="price")


def test_add_terminal_rejects_name_that_matches_an_argument():
    pset = gp.PrimitiveSetTyped("main", [float], float)
    pset.rename_arguments(ARG0="price")
    with pytest.raises(ValueError, match="shadow"):
        pset.add_terminal(1.0, float, name="price")

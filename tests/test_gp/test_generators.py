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


def test_generate_raises_when_no_terminal_or_primitive_exists():
    pset = gp.PrimitiveSetTyped("main", [], float)
    with pytest.raises(IndexError, match="terminal"):
        gp.generate(pset, 0, 0, lambda _height, _depth: True, float)
    with pytest.raises(IndexError, match="primitive"):
        gp.generate(pset, 1, 1, lambda _height, _depth: False, float)


def test_generate_instantiates_ephemeral_classes():
    pset = gp.PrimitiveSet("main", 0)
    pset.add_ephemeral_constant("COV_EPH_GEN", lambda: 7)
    tree = gp.generate(pset, 0, 0, lambda _height, _depth: True)
    assert tree[0].value == 7
    mixed = gp.PrimitiveSet("main", 1)
    mixed.add_primitive(operator.add, 2)
    assert gp.gen_half_and_half(mixed, 1, 2)

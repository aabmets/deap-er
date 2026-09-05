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
from deap_er import gp, tools
from deap_er.private.programming.generators import choose_weighted


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


def test_choose_weighted_keeps_unweighted_rng_choice():
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.sub, 2)
    prims = pset.primitives[object]
    tools.rng.seed(5)
    picked = choose_weighted(prims)
    tools.rng.seed(5)
    assert picked is tools.rng.choice(prims)


def test_weighted_primitives_skew_generation():
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2, weight=1000.0)
    pset.add_primitive(operator.sub, 2, weight=0.001)
    tools.rng.seed(1)
    names = [gp.gen_full(pset, 1, 1)[0].name for _ in range(40)]
    assert names.count("add") > names.count("sub")

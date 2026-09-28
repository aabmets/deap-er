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
from unittest.mock import patch

import numpy
from deap_er import Fitness, Toolbox, creator, gp, tools
from deap_er.private.programming.crossover import index_at_path, path_from_root
from deap_er.private.programming.tape_lower import child_indices


def _assert_well_typed(tree: gp.PrimitiveTree) -> None:
    stack: list[type] = []
    for node in reversed(list(tree)):
        if node.arity == 0:
            stack.append(node.ret)
            continue
        args = [stack.pop() for _ in range(node.arity)]
        assert args == list(node.args)
        stack.append(node.ret)


def _typed_object_pset():
    def wrap_int(value: int) -> object:
        return value

    def wrap_str(value: str) -> object:
        return value

    def concat(left: str, right: str) -> str:
        return left + right

    pset = gp.PrimitiveSetTyped("main", [], object)
    pset.add_primitive(wrap_int, [int], object, name="wrap_int")
    pset.add_primitive(wrap_str, [str], object, name="wrap_str")
    pset.add_primitive(operator.add, [int, int], int, name="add")
    pset.add_primitive(concat, [str, str], str, name="cat")
    pset.add_terminal(1, int, name="one")
    pset.add_terminal("x", str, name="ex")
    return pset


def test_homologous_swaps_same_path_when_types_match():
    pset = _typed_object_pset()
    pset.add_terminal(2, int, name="two")
    changed = False
    for seed in range(30):
        tools.rng.seed(seed)
        first: Any = gp.PrimitiveTree.from_string("wrap_int(add(one, one))", pset)
        second: Any = gp.PrimitiveTree.from_string("wrap_int(add(one, two))", pset)
        before_first = str(first)
        before_second = str(second)
        gp.cx_homologous(first, second)
        if str(first) != before_first or str(second) != before_second:
            changed = True
            break
    assert changed


def test_homologous_falls_back_when_path_missing():
    pset = _typed_object_pset()
    for seed in range(40):
        tools.rng.seed(seed)
        first: Any = gp.PrimitiveTree.from_string("wrap_int(add(one, one))", pset)
        second: Any = gp.PrimitiveTree.from_string("wrap_str(cat(ex, ex))", pset)
        gp.cx_homologous(first, second)
        _assert_well_typed(first)
        _assert_well_typed(second)


def test_homologous_short_trees_unchanged():
    pset = _typed_object_pset()
    first: Any = gp.PrimitiveTree.from_string("one", pset)
    second: Any = gp.PrimitiveTree.from_string("ex", pset)
    before_first = list(first)
    before_second = list(second)
    gp.cx_homologous(first, second)
    assert list(first) == before_first
    assert list(second) == before_second


def test_public_cx_semantic_is_distinct_from_cx_one_point_semantic():
    assert gp.cx_semantic is not gp.cx_one_point_semantic
    assert gp.cx_homologous is not gp.cx_one_point
    assert gp.cx_homologous is not gp.cx_semantic


def test_register_gp_keeps_cx_one_point_as_default_mate():
    creator.create_type("REG39_FIT", Fitness, weights=(-1.0,))
    creator.create_type(
        "REG39_IND",
        gp.PrimitiveTree,
        fitness=creator.__dict__["REG39_FIT"],
    )
    try:
        toolbox = Toolbox()
        pset = gp.PrimitiveSet("MAIN", 1)
        pset.add_primitive(operator.add, 2)
        pset.add_terminal(1.0)
        gp.register_gp(
            toolbox,
            pset,
            individual=creator.__dict__["REG39_IND"],
            height_limit=None,
            select=False,
        )
        assert toolbox.mate.func is gp.cx_one_point
    finally:
        del creator.__dict__["REG39_FIT"]
        del creator.__dict__["REG39_IND"]


def test_homologous_type_mismatch_at_path_skips_direct_swap():
    pset = _typed_object_pset()
    ind1: Any = gp.PrimitiveTree.from_string("wrap_int(add(one, one))", pset)
    ind2: Any = gp.PrimitiveTree.from_string("wrap_str(cat(ex, ex))", pset)
    before_first = str(ind1)
    before_second = str(ind2)
    with patch(
        "deap_er.private.programming.crossover.rng.integers",
        return_value=numpy.int64(1),
    ):
        gp.cx_homologous(ind1, ind2)
    assert str(ind1) == before_first
    assert str(ind2) == before_second
    _assert_well_typed(ind1)
    _assert_well_typed(ind2)


def test_homologous_matching_path_swaps_when_anchor_forced():
    pset = _typed_object_pset()
    pset.add_terminal(2, int, name="two")
    ind1: Any = gp.PrimitiveTree.from_string("wrap_int(add(one, one))", pset)
    ind2: Any = gp.PrimitiveTree.from_string("wrap_int(add(one, two))", pset)
    before_first = str(ind1)
    before_second = str(ind2)
    with patch(
        "deap_er.private.programming.crossover.rng.integers",
        return_value=numpy.int64(1),
    ):
        gp.cx_homologous(ind1, ind2)
    assert str(ind1) != before_first or str(ind2) != before_second


def test_homologous_invalid_path_uses_fallback_not_direct_swap():
    pset = _typed_object_pset()
    ind1: Any = gp.PrimitiveTree.from_string("wrap_int(add(one, add(one, one)))", pset)
    ind2: Any = gp.PrimitiveTree.from_string("wrap_int(one)", pset)

    index1 = 3
    path = path_from_root(child_indices(list(ind1)), index1)
    assert index_at_path(child_indices(list(ind2)), path) is None

    before_first = str(ind1)
    before_second = str(ind2)
    with patch(
        "deap_er.private.programming.crossover.rng.integers",
        return_value=numpy.int64(index1),
    ):
        gp.cx_homologous(ind1, ind2)
    assert str(ind1) != before_first or str(ind2) != before_second
    _assert_well_typed(ind1)
    _assert_well_typed(ind2)

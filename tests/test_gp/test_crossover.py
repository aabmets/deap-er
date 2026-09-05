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

from deap_er.gp.crossover import cx_one_point
from deap_er.gp.primitives import PrimitiveSetTyped, PrimitiveTree
from deap_er.rng import rng


def _assert_well_typed(tree: PrimitiveTree) -> None:
    stack: list[type] = []
    for node in reversed(list(tree)):
        if node.arity == 0:
            stack.append(node.ret)
            continue
        args = [stack.pop() for _ in range(node.arity)]
        assert args == list(node.args)
        stack.append(node.ret)


def test_typed_object_root_does_not_swap_incompatible_subtrees():
    def wrap_int(value: int) -> object:
        return value

    def wrap_str(value: str) -> object:
        return value

    def concat(left: str, right: str) -> str:
        return left + right

    pset = PrimitiveSetTyped("main", [], object)
    pset.add_primitive(wrap_int, [int], object, name="wrap_int")
    pset.add_primitive(wrap_str, [str], object, name="wrap_str")
    pset.add_primitive(operator.add, [int, int], int, name="add")
    pset.add_primitive(concat, [str, str], str, name="cat")
    pset.add_terminal(1, int, name="one")
    pset.add_terminal("x", str, name="ex")

    for seed in range(40):
        rng.seed(seed)
        first: Any = PrimitiveTree.from_string("wrap_int(add(one, one))", pset)
        second: Any = PrimitiveTree.from_string("wrap_str(cat(ex, ex))", pset)
        cx_one_point(first, second)
        _assert_well_typed(first)
        _assert_well_typed(second)

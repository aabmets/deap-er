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

from deap_er import gp


def test_zero_arity_callable_terminal_formats_as_call():
    def seven() -> int:
        return 7

    pset = gp.PrimitiveSet("main", 0)
    pset.add_terminal(seven, call_zero=True)
    pset.add_primitive(operator.add, 2)
    tree = gp.PrimitiveTree.from_string("add(seven, seven)", pset)
    assert "seven()" in str(tree)
    assert gp.compile_tree(tree, pset) == 14
    assert gp.tree_to_infix(tree) == "(seven() + seven())"


def test_callable_action_terminals_stay_function_objects():
    calls: list[int] = []

    def turn() -> None:
        calls.append(1)

    def wrap(left: object, right: object) -> tuple[object, object]:
        return left, right

    pset = gp.PrimitiveSet("main", 0)
    pset.add_primitive(wrap, 2)
    pset.add_terminal(turn)
    tree = gp.PrimitiveTree.from_string("wrap(turn, turn)", pset)
    assert "turn()" not in str(tree)
    assert gp.compile_tree(tree, pset) == (turn, turn)
    assert calls == []


def test_argument_terminals_are_not_wrapped_as_calls():
    pset = gp.PrimitiveSet("main", 1)
    tree = gp.PrimitiveTree.from_string("ARG0", pset)
    assert str(tree) == "ARG0"
    assert gp.compile_tree(tree, pset)(9) == 9

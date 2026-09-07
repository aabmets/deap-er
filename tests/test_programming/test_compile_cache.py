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
from deap_er.private.programming.compile_cache import CompileCache, expression_key
from deap_er.private.programming.compilers import _compile_cache, invalidate_compiled


def test_compile_cache_evicts_the_oldest_entry():
    cache = CompileCache(maxsize=2)
    cache.set(("a",), 1)
    cache.set(("b",), 2)
    cache.set(("c",), 3)

    assert cache.get(("a",)) is None
    assert cache.get(("b",)) == 2
    assert cache.get(("c",)) == 3


def test_compile_cache_touch_keeps_a_hot_entry():
    cache = CompileCache(maxsize=2)
    cache.set(("a",), 1)
    cache.set(("b",), 2)
    cache.get(("a",))
    cache.set(("c",), 3)

    assert cache.get(("a",)) == 1
    assert cache.get(("b",)) is None
    assert cache.get(("c",)) == 3


def test_compile_tree_lru_does_not_flush_the_whole_cache():
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    small = CompileCache(maxsize=3)
    previous = _compile_cache
    try:
        import deap_er.private.programming.compilers as compilers

        compilers._compile_cache = small
        for value in range(5):
            tree = gp.PrimitiveTree.from_string(f"add(ARG0, {value})", pset)
            gp.compile_tree(tree, pset)
        assert len(small) == 3
    finally:
        compilers._compile_cache = previous


def test_expression_key_uses_nodes_for_trees_and_text_for_source():
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    tree = gp.PrimitiveTree.from_string("add(ARG0, 2)", pset)

    assert expression_key("add(ARG0, 2)") == "add(ARG0, 2)"
    assert expression_key(tree) == (
        ("add", None, 2, False),
        ("ARG0", "ARG0", 0, False),
        ("2", 2, 0, False),
    )


def test_discard_expression_drops_structural_tree_keys():
    cache = CompileCache()
    marker = object()
    struct = (("add", None, 2, False), ("ARG0", "ARG0", 0, False), ("2", 2, 0, False))
    cache.set(("python", 0, struct, ("ARG0",), (), 0), marker)
    cache.set(("python", 0, "add(ARG0, 2)", ("ARG0",), (), 0), marker)
    cache.set(("opcode", 0, "mul(1, 2)", (), (), 0), marker)

    assert cache.discard_expression(struct) == 1
    assert cache.discard_expression("add(ARG0, 2)") == 1
    assert cache.get(("opcode", 0, "mul(1, 2)", (), (), 0)) is marker


def test_invalidate_compiled_drops_a_live_tree_entry():
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    tree = gp.PrimitiveTree.from_string("add(ARG0, 2)", pset)
    compiled = gp.compile_tree(tree, pset)

    assert gp.compile_tree(tree, pset) is compiled
    assert invalidate_compiled(tree) == 1
    assert gp.compile_tree(tree, pset) is not compiled


def test_compile_tree_zero_and_one_arg_do_not_share_cache():
    shared_add = operator.add
    zero = gp.PrimitiveSet("zero", 0)
    zero.add_terminal(2, name="two")
    zero.add_terminal(3, name="three")
    zero.add_primitive(shared_add, 2)
    one = gp.PrimitiveSet("one", 1)
    one.add_terminal(2, name="two")
    one.add_terminal(3, name="three")
    one.add_primitive(shared_add, 2)
    source = "add(two, three)"

    assert gp.compile_tree(source, zero) == 5
    func = gp.compile_tree(source, one)
    assert func(0) == 5
    assert gp.compile_tree(source, zero) == 5
    assert gp.compile_tree(source, one) is func


def test_expression_key_falls_back_when_a_tree_is_not_hashable():
    class Leaf:
        name = "x"
        value = ["unhashable"]
        arity = 0
        call_zero = False

    nodes = [Leaf()]
    assert expression_key(nodes) == str(nodes)

    pset = gp.PrimitiveSet("main", 1)
    slim = gp.SlimTree([pset.mapping["ARG0"]])
    assert expression_key(slim) == str(slim)

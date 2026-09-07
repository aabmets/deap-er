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
from deap_er.private.programming.compile_cache import CompileCache
from deap_er.private.programming.compilers import _compile_cache


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

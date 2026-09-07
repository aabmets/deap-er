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
from deap_er import gp, tools
from deap_er.private.programming.compilers import _compile_cache
from deap_er.private.programming.promote import PromotedLibrary


def _typed_set():
    pset = gp.PrimitiveSetTyped("main", [float, float], float)
    pset.add_primitive(operator.add, [float, float], float)
    pset.add_primitive(operator.mul, [float, float], float)
    return pset


def test_promote_subtree_registers_types_and_is_sampled():
    pset = _typed_set()
    tree = gp.PrimitiveTree.from_string("add(ARG0, mul(ARG1, ARG0))", pset)
    name = gp.promote_subtree(pset, tree, weight=1000.0)
    prim = pset.mapping[name]
    assert name == "promo0"
    assert gp.promoted_names(pset) == [name]
    assert prim.args == [float, float]
    assert prim.ret is float
    compiled = gp.compile_tree(gp.PrimitiveTree.from_string(f"{name}(ARG0, ARG1)", pset), pset)
    assert compiled(3.0, 4.0) == 3.0 + 4.0 * 3.0
    tools.rng.seed(1)
    found = any(
        any(getattr(node, "name", None) == name for node in gp.gen_full(pset, 1, 2))
        for _ in range(40)
    )
    assert found


def test_promote_inner_subtree_and_zero_arity_constant():
    pset = _typed_set()
    tree = gp.PrimitiveTree.from_string("add(ARG0, mul(ARG1, ARG0))", pset)
    name = gp.promote_subtree(pset, tree, index=2)
    assert pset.mapping[name].args == [float, float]
    pset.add_terminal(1.0, float, name="one")
    const = gp.promote_subtree(pset, gp.PrimitiveTree.from_string("add(one, one)", pset))
    assert pset.mapping[const].arity == 0
    func = gp.compile_tree(gp.PrimitiveTree.from_string(f"{const}()", pset), pset)
    assert func(9.0, 0.0) == 2.0


def test_promote_rejects_incomplete_ill_typed_and_trivial_trees():
    pset = _typed_set()
    tree = gp.PrimitiveTree.from_string("add(ARG0, ARG1)", pset)
    with pytest.raises(ValueError, match="complete"):
        gp.promote_subtree(pset, [tree[0]])
    with pytest.raises(ValueError, match="complete"):
        gp.promote_subtree(pset, list(tree) + [tree[1]])
    bad = [pset.mapping["add"], pset.mapping["ARG0"], gp.Terminal(True, False, bool)]
    with pytest.raises(TypeError, match="return type"):
        gp.promote_subtree(pset, bad)
    with pytest.raises(ValueError, match="lone argument"):
        gp.promote_subtree(pset, [pset.mapping["ARG0"]])
    with pytest.raises(IndexError, match="outside"):
        gp.promote_subtree(pset, tree, index=9)
    with pytest.raises(ValueError, match="max_library"):
        gp.promote_subtree(pset, tree, max_library=0)


def test_promote_rejects_adf_and_untyped_zero_arity():
    adf = gp.PrimitiveSet("ADF0", 2)
    adf.add_primitive(operator.add, 2)
    main = gp.PrimitiveSet("MAIN", 1)
    main.add_adf(adf)
    main.add_primitive(operator.add, 2)
    adf_tree = gp.PrimitiveTree.from_string("ADF0(ARG0, ARG0)", main)
    with pytest.raises(ValueError, match="ADF"):
        gp.promote_subtree(main, adf_tree)
    loose = gp.PrimitiveSet("main", 1)
    loose.add_primitive(operator.add, 2)
    loose.add_terminal(1, name="one")
    const_tree = gp.PrimitiveTree.from_string("add(one, one)", loose)
    with pytest.raises(ValueError, match="arity"):
        gp.promote_subtree(loose, const_tree)


def test_add_adf_still_compiles_after_promote():
    adf = gp.PrimitiveSet("ADF0", 2)
    adf.add_primitive(operator.add, 2)
    main = gp.PrimitiveSet("MAIN", 1)
    main.add_adf(adf)
    main.add_primitive(operator.mul, 2)
    gp.promote_subtree(main, gp.PrimitiveTree.from_string("mul(ARG0, ARG0)", main))
    expressions: Any = [
        gp.PrimitiveTree.from_string("ADF0(ARG0, ARG0)", main),
        gp.PrimitiveTree.from_string("add(ARG0, ARG1)", adf),
    ]
    func = gp.compile_adf_tree(expressions, [main, adf])
    assert func(3) == 6


def test_library_cap_evicts_least_used_promoted_name_only():
    pset = _typed_set()
    first = gp.promote_subtree(
        pset, gp.PrimitiveTree.from_string("add(ARG0, ARG1)", pset), weight=1000.0
    )
    tools.rng.seed(2)
    for _ in range(20):
        gp.gen_full(pset, 1, 1)
    second = gp.promote_subtree(pset, gp.PrimitiveTree.from_string("mul(ARG0, ARG1)", pset))
    third = gp.promote_subtree(
        pset, gp.PrimitiveTree.from_string("add(ARG0, ARG0)", pset), max_library=2
    )
    names = gp.promoted_names(pset)
    assert first in names
    assert second not in names
    assert third in names
    assert "add" in pset.mapping
    assert "mul" in pset.mapping
    assert second not in pset.context
    fourth = gp.promote_subtree(pset, gp.PrimitiveTree.from_string("mul(ARG1, ARG1)", pset))
    assert fourth == "promo3"


def test_colliding_generated_name_is_skipped():
    pset = _typed_set()
    pset.add_primitive(operator.neg, [float], float, name="promo0")
    name = gp.promote_subtree(pset, gp.PrimitiveTree.from_string("add(ARG0, ARG1)", pset))
    assert name == "promo1"


def test_promote_and_clear_compile_cache_drop_entries():
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    tree = gp.PrimitiveTree.from_string("add(ARG0, 1)", pset)
    gp.compile_tree(tree, pset)
    assert len(_compile_cache) > 0
    name = gp.promote_subtree(pset, gp.PrimitiveTree.from_string("add(ARG0, ARG0)", pset))
    assert len(_compile_cache) == 0
    promo = gp.PrimitiveTree.from_string(f"{name}(ARG0)", pset)
    old_fn = gp.compile_tree(promo, pset)
    assert old_fn(4) == 8
    gp.promote_subtree(pset, tree, max_library=1)
    assert name not in pset.context
    with pytest.raises(NameError, match=name):
        gp.compile_tree(promo, pset)
    gp.compile_tree(tree, pset)
    gp.clear_compile_cache()
    assert len(_compile_cache) == 0


def test_failed_promote_rolls_back_eviction():
    pset = _typed_set()
    first = gp.promote_subtree(pset, gp.PrimitiveTree.from_string("add(ARG0, ARG1)", pset))
    bound = dict(gp.numba_opcodes())
    second = gp.PrimitiveTree.from_string("mul(ARG0, ARG1)", pset)
    with pytest.raises(ValueError, match="weight"):
        gp.promote_subtree(pset, second, max_library=1, weight=0.0)
    assert gp.promoted_names(pset) == [first]
    assert first in pset.mapping
    assert first in pset.context
    assert gp.numba_opcodes() == bound


def test_mutation_increments_promoted_use():
    pset = _typed_set()
    name = gp.promote_subtree(
        pset, gp.PrimitiveTree.from_string("add(ARG0, ARG1)", pset), weight=1000.0
    )
    parent = gp.PrimitiveTree.from_string("add(ARG0, ARG1)", pset)
    tools.rng.seed(3)
    for _ in range(30):
        mutant: Any = gp.PrimitiveTree(list(parent))
        gp.mut_insert(mutant, pset)
    library = pset.promoted_library
    assert isinstance(library, PromotedLibrary)
    assert library.records[name].uses > 0
    assert gp.promoted_names(gp.PrimitiveSetTyped("empty", [float], float)) == []

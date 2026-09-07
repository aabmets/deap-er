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
import numpy
import pytest
from deap_er import gp


def _column_set():
    pset = gp.make_column_pset(["x", "y"])
    gp.add_numpy_primitives(pset)
    return pset


def _samples(rows=24):
    generator = numpy.random.default_rng(11)
    return generator.normal(size=rows), generator.normal(size=rows) * 2.0


def test_promote_binds_user_opcode_and_expands_on_lower():
    pset = _column_set()
    tree = gp.PrimitiveTree.from_string("vadd(x, vmul(y, x))", pset)
    name = gp.promote_subtree(pset, tree)
    prim = pset.mapping[name]
    assert prim.args == [gp.Array, gp.Array]
    assert prim.ret is gp.Array
    assert gp.numba_opcodes()[name] >= gp.USER_BASE
    call = gp.PrimitiveTree.from_string(f"{name}(x, y)", pset)
    tape = gp.lower_tree(call, pset)
    assert not numpy.any(tape.opcodes >= gp.USER_BASE)


def test_promoted_columnar_backends_agree():
    pset = _column_set()
    name = gp.promote_subtree(pset, gp.PrimitiveTree.from_string("vadd(x, vmul(y, x))", pset))
    tree = gp.PrimitiveTree.from_string(f"{name}(x, y)", pset)
    first, second = _samples()
    expected = first + second * first
    python = gp.compile_tree(tree, pset)(first, second)
    opcode = gp.compile_tree(tree, pset, backend="opcode")(first, second)
    numpy.testing.assert_allclose(python, expected)
    numpy.testing.assert_allclose(opcode, expected)
    matrix = numpy.column_stack([first, second])
    batch = gp.interpret_tapes([gp.lower_tree(tree, pset)], matrix)
    numpy.testing.assert_allclose(batch[0], expected)
    if gp.numba_available():
        numba = gp.compile_tree(tree, pset, backend="numba")(first, second)
        numpy.testing.assert_allclose(numba, expected)


def test_nested_promote_expands_fully():
    pset = _column_set()
    inner = gp.promote_subtree(pset, gp.PrimitiveTree.from_string("vadd(x, y)", pset))
    outer = gp.promote_subtree(pset, gp.PrimitiveTree.from_string(f"vmul({inner}(x, y), x)", pset))
    tree = gp.PrimitiveTree.from_string(f"{outer}(x, y)", pset)
    tape = gp.lower_tree(tree, pset)
    assert not numpy.any(tape.opcodes >= gp.USER_BASE)
    first, second = _samples()
    expected = (first + second) * first
    numpy.testing.assert_allclose(gp.compile_tree(tree, pset)(first, second), expected)
    numpy.testing.assert_allclose(
        gp.compile_tree(tree, pset, backend="opcode")(first, second), expected
    )


def test_window_subtree_bakes_the_length():
    pset = gp.make_column_pset(["x"])
    gp.add_window_primitives(pset)
    gp.add_window_ephemeral(pset, "win", 3, 3)
    window = pset.terminals[gp.Window][0]()
    tree = gp.PrimitiveTree([pset.mapping["rolling_mean"], pset.mapping["x"], window])
    name = gp.promote_subtree(pset, tree)
    assert pset.mapping[name].args == [gp.Array]
    assert pset.mapping[name].ret is gp.Array
    call = gp.PrimitiveTree.from_string(f"{name}(x)", pset)
    column = numpy.arange(8, dtype=numpy.float64)
    expected = gp.rolling_mean(column, 3)
    numpy.testing.assert_allclose(gp.compile_tree(call, pset)(column), expected)
    numpy.testing.assert_allclose(gp.compile_tree(call, pset, backend="opcode")(column), expected)
    tape = gp.lower_tree(call, pset)
    assert not numpy.any(tape.opcodes >= gp.USER_BASE)


def test_evict_inner_keeps_outer_tape_on_builtins():
    pset = _column_set()
    inner = gp.promote_subtree(pset, gp.PrimitiveTree.from_string("vadd(x, y)", pset))
    outer = gp.promote_subtree(pset, gp.PrimitiveTree.from_string(f"vmul({inner}(x, y), x)", pset))
    gp.promote_subtree(pset, gp.PrimitiveTree.from_string("vsub(x, y)", pset), max_library=2)
    assert inner not in gp.promoted_names(pset)
    assert outer in gp.promoted_names(pset)
    tree = gp.PrimitiveTree.from_string(f"{outer}(x, y)", pset)
    tape = gp.lower_tree(tree, pset)
    assert not numpy.any(tape.opcodes >= gp.USER_BASE)
    first, second = _samples()
    expected = (first + second) * first
    numpy.testing.assert_allclose(gp.compile_tree(tree, pset)(first, second), expected)
    numpy.testing.assert_allclose(
        gp.compile_tree(tree, pset, backend="opcode")(first, second), expected
    )


def test_failed_columnar_promote_does_not_orphan_opcode():
    pset = _column_set()
    first = gp.promote_subtree(pset, gp.PrimitiveTree.from_string("vadd(x, y)", pset))
    bound = dict(gp.numba_opcodes())
    second = gp.PrimitiveTree.from_string("vmul(x, y)", pset)
    with pytest.raises(ValueError, match="weight"):
        gp.promote_subtree(pset, second, max_library=1, weight=0.0)
    assert gp.promoted_names(pset) == [first]
    assert gp.numba_opcodes() == bound


def test_promote_rejects_a_window_that_is_not_a_leaf():
    pset = gp.make_column_pset(["x"])
    gp.add_window_primitives(pset)
    gp.add_window_ephemeral(pset, "win", 3, 3)
    pset.add_primitive(min, [gp.Window, gp.Window], gp.Window, "wmin")
    window = pset.terminals[gp.Window][0]
    tree = gp.PrimitiveTree(
        [pset.mapping["rolling_mean"], pset.mapping["x"], pset.mapping["wmin"], window(), window()]
    )
    with pytest.raises(ValueError, match="must be a leaf"):
        gp.promote_subtree(pset, tree)


def test_eviction_keeps_builtin_columnar_primitives():
    pset = _column_set()
    first = gp.promote_subtree(pset, gp.PrimitiveTree.from_string("vadd(x, y)", pset))
    gp.promote_subtree(pset, gp.PrimitiveTree.from_string("vmul(x, y)", pset), max_library=1)
    assert first not in gp.promoted_names(pset)
    assert "vadd" in pset.mapping
    assert "vmul" in pset.mapping
    assert "vadd" in gp.BUILTIN_OPCODES

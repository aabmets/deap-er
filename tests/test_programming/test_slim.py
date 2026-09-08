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
import math
import operator
from typing import Any

from deap_er import gp, tools


def lf(x):
    return 1 / (1 + math.exp(-x))


def _semantic_pset():
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.sub, 2)
    pset.add_primitive(operator.mul, 2)
    pset.add_primitive(lf, 1, name="lf")
    pset.add_terminal(1.0)
    pset.rename_arguments(ARG0="x")
    return pset


def _slim_head(pset):
    return gp.SlimTree.from_tree(gp.gen_grow(pset, 1, 2))


def test_slim_inflate_appends_block():
    tools.rng.seed(0)
    pset = _semantic_pset()
    head = gp.gen_grow(pset, 1, 2)
    slim = gp.SlimTree(gp.PrimitiveTree(head))
    head_len = len(head)
    (mutated,) = gp.mut_slim_inflate(slim, pset, min_depth=1, max_depth=1, mut_step=0.5)
    assert len(mutated.deltas) == 1
    assert len(mutated.head) == head_len
    assert len(mutated) > head_len


def test_slim_deflate_removes_block():
    tools.rng.seed(1)
    pset = _semantic_pset()
    slim = _slim_head(pset)
    gp.mut_slim_inflate(slim, pset, min_depth=1, max_depth=1, mut_step=0.5)
    gp.mut_slim_inflate(slim, pset, min_depth=1, max_depth=1, mut_step=0.5)
    size_before = len(slim)
    (mutated,) = gp.mut_slim_deflate(slim)
    assert len(mutated.deltas) == 1
    assert len(mutated) < size_before


def test_slim_deflate_noop_without_deltas():
    tools.rng.seed(2)
    pset = _semantic_pset()
    slim = _slim_head(pset)
    (mutated,) = gp.mut_slim_deflate(slim)
    assert mutated is slim
    assert mutated.deltas == []


def test_compile_slim_tree_semantics():
    tools.rng.seed(3)
    pset = _semantic_pset()
    slim = _slim_head(pset)
    gp.mut_slim_inflate(slim, pset, min_depth=1, max_depth=1, mut_step=0.25)
    compiled = gp.compile_slim_tree(slim, pset)
    head_fn = gp.compile_tree(slim.head, pset)
    delta_fn = gp.compile_tree(slim.deltas[0], pset)
    for x in (-1.0, 0.0, 0.5, 1.0):
        assert math.isclose(compiled(x), head_fn(x) + delta_fn(x), rel_tol=0, abs_tol=1e-12)


def test_inflate_is_ball_perturbation():
    tools.rng.seed(4)
    pset = _semantic_pset()
    slim = _slim_head(pset)
    mut_step = 0.4
    before = gp.compile_slim_tree(slim, pset)
    gp.mut_slim_inflate(slim, pset, min_depth=1, max_depth=1, mut_step=mut_step)
    after = gp.compile_slim_tree(slim, pset)
    for x in (-2.0, -0.5, 0.0, 0.5, 2.0):
        assert abs(after(x) - before(x)) <= mut_step + 1e-9


def test_deflate_never_removes_head():
    tools.rng.seed(9)
    pset = _semantic_pset()
    slim = _slim_head(pset)
    head_snapshot = list(slim.head)
    for step in range(3):
        gp.mut_slim_inflate(slim, pset, min_depth=1, max_depth=1, mut_step=0.1 * (step + 1))
        gp.mut_slim_deflate(slim)
    assert [node.name for node in slim.head] == [node.name for node in head_snapshot]


def test_slim_inflate_mutates_in_place():
    tools.rng.seed(11)
    pset = _semantic_pset()
    slim = _slim_head(pset)
    (mutated,) = gp.mut_slim_inflate(slim, pset, min_depth=1, max_depth=1, mut_step=0.2)
    assert mutated is slim


def test_mut_slim_inflate_rejects_primitive_tree():
    pset = _semantic_pset()
    tree: Any = gp.gen_grow(pset, 1, 2)
    try:
        gp.mut_slim_inflate(tree, pset, min_depth=1, max_depth=1, mut_step=0.2)
    except TypeError as err:
        assert "SlimTree" in str(err)
    else:
        raise AssertionError("expected TypeError for PrimitiveTree input")


def test_slim_deflate_without_semantic_primitives():
    tools.rng.seed(16)
    semantic = _semantic_pset()
    slim = _slim_head(semantic)
    gp.mut_slim_inflate(slim, semantic, min_depth=1, max_depth=1, mut_step=0.2)
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_terminal(1.0)
    pset.rename_arguments(ARG0="x")
    assert len(slim.deltas) == 1
    (mutated,) = gp.mut_slim_deflate(slim)
    assert mutated is slim
    assert mutated.deltas == []

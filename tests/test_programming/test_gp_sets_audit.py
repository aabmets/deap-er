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
import copy
import functools
import operator
import pickle
from typing import Any, cast

import dill
import pytest
from deap_er import Checkpoint, gp, tools
from deap_er.private.programming.numpy.numpy_ops import infer_fill


def _draw_three() -> float:
    return 3.0


def _typed_set() -> gp.PrimitiveSetTyped:
    pset = gp.PrimitiveSetTyped("main", [float], float)
    pset.add_primitive(operator.add, [float, float], float, name="add")
    pset.add_ephemeral_constant("audit_eph_three", _draw_three, float)
    return pset


def _ephemeral_tree(pset: gp.PrimitiveSetTyped) -> gp.PrimitiveTree:
    eph_cls = next(t for t in pset.terminals[float] if isinstance(t, type))
    return gp.PrimitiveTree([pset.mapping["add"], pset.mapping["ARG0"], eph_cls()])


def _window_tree() -> gp.PrimitiveTree:
    pset = gp.columnar_pset(["level"], window=(2, 9), window_name="audit_window_2_9")
    win_cls = next(t for t in pset.terminals[gp.Window] if isinstance(t, type))
    delay = pset.mapping["delay"]
    leaves = {gp.Window: win_cls(), delay.args[0]: pset.mapping["level"]}
    return gp.PrimitiveTree([delay, *(leaves[arg] for arg in delay.args)])


# ---- S2b-D1: ephemeral classes pickle with the standard library ----


def test_tree_with_ephemeral_round_trips_through_stdlib_pickle():
    tree = _ephemeral_tree(_typed_set())

    restored = pickle.loads(pickle.dumps(tree))

    assert str(restored) == str(tree)
    assert type(restored[2]) is type(tree[2])


def test_tree_with_window_ephemeral_round_trips_through_stdlib_pickle():
    tree = _window_tree()

    restored = pickle.loads(pickle.dumps(tree))

    assert str(restored) == str(tree)
    assert type(restored[2]) is type(tree[2])
    assert restored[2].ret is gp.Window


# ---- S1b-D5: nodes are read-only after construction ----


def test_assigning_a_primitive_attribute_raises_and_leaves_the_set_alone():
    pset = _typed_set()
    tree = gp.PrimitiveTree.from_string("add(ARG0, ARG0)", pset)
    before = pset.mapping["add"].seq

    with pytest.raises(AttributeError, match="read-only"):
        tree[0].seq = "sub({0}, {1})"
    with pytest.raises(AttributeError, match="read-only"):
        tree[0].name = "sub"
    with pytest.raises(AttributeError, match="read-only"):
        del tree[0].args

    assert pset.mapping["add"].seq == before
    assert str(tree) == "add(ARG0, ARG0)"


def test_assigning_a_terminal_or_ephemeral_attribute_raises():
    tree = _ephemeral_tree(_typed_set())

    with pytest.raises(AttributeError, match="read-only"):
        tree[1].value = "ARG9"
    with pytest.raises(AttributeError, match="read-only"):
        tree[2].value = 7.0
    with pytest.raises(AttributeError, match="read-only"):
        tree[2].ret = int

    assert tree[1].value == "ARG0"
    assert tree[2].value == 3.0


def test_read_only_nodes_survive_copy_deepcopy_pickle_and_dill():
    tree = _ephemeral_tree(_typed_set())

    for clone in (
        copy.deepcopy(tree),
        pickle.loads(pickle.dumps(tree)),
        dill.loads(dill.dumps(tree)),
    ):
        assert str(clone) == str(tree)
    for node in tree:
        assert copy.copy(node) == node
        assert copy.deepcopy(node) == node
    with pytest.raises(AttributeError, match="read-only"):
        copy.copy(tree[2]).value = 1.0


def test_read_only_nodes_survive_a_checkpoint(tmp_path):
    tree = _window_tree()
    saved = Checkpoint(file_name="nodes.cpt", dir_path=tmp_path, autoload=False)
    saved.tree = tree
    saved.save()

    loaded = Checkpoint(file_name="nodes.cpt", dir_path=tmp_path, autoload=False)
    loaded.load()

    assert str(loaded.tree) == str(tree)


def test_gp_operators_and_renames_still_work_on_read_only_nodes():
    pset = _typed_set()
    tools.rng.seed(3)
    tree = gp.PrimitiveTree(gp.gen_full(pset, 2, 3))

    (mutant,) = gp.mut_ephemeral(cast(Any, copy.deepcopy(tree)), mode="all")
    pset.rename_arguments(ARG0="x")
    renamed = gp.PrimitiveTree.from_string("add(x, x)", pset)

    assert len(mutant) == len(tree)
    assert str(renamed) == "add(x, x)"


# ---- S1b-D6: columnar_pset forwards a protection fill ----


def test_columnar_pset_binds_the_requested_fill():
    pset = gp.columnar_pset(["level"], window=None, ema=False, fill=-2.5)

    assert infer_fill(pset) == -2.5


# ---- S1a-D3: static_limit clones only the individuals ----


class _CopySpy:
    copies = 0

    def __deepcopy__(self, memo: dict[int, Any]) -> "_CopySpy":
        type(self).copies += 1
        return _CopySpy()


def test_static_limit_does_not_clone_a_positionally_bound_object():
    def bloat(individual: list[int], _prim_set: _CopySpy) -> tuple[list[int]]:
        return (individual + [0] * 10,)

    spy = _CopySpy()
    limited = gp.static_limit(len, 3)(bloat)
    for seed in range(20):
        tools.rng.seed(seed)
        (child,) = limited([1], spy)
        assert child == [1]

    assert _CopySpy.copies == 0


def test_static_limit_with_positional_prim_set_returns_only_trees():
    pset = _typed_set()
    tools.rng.seed(5)
    parent = gp.PrimitiveTree(gp.gen_full(pset, 1, 1))
    limited = gp.static_limit(operator.attrgetter("height"), 1)(gp.mut_uniform)

    expr = functools.partial(gp.gen_full, min_depth=3, max_depth=3)

    for seed in range(20):
        tools.rng.seed(seed)
        (child,) = limited(copy.deepcopy(parent), expr, pset)
        assert isinstance(child, gp.PrimitiveTree)
        assert child.height <= 1

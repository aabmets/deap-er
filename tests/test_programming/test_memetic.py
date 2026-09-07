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

import numpy
import pytest
from deap_er import Fitness, creator, gp, tools
from deap_er.private.programming.compile_cache import CompileCache
from deap_er.private.programming.ephemeral_leaves import leaf_range

FIT = "MEMETIC_FIT"
IND = "MEMETIC_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, gp.PrimitiveTree, fitness=creator.__dict__[FIT])
    yield creator.__dict__[IND]
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def _float_pset(name: str) -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_ephemeral_constant(name, lambda: 0.25)
    return pset


def _two_leaf_tree(ind_cls, pset: gp.PrimitiveSet):
    eph = pset.terminals[object][-1]
    return ind_cls([pset.mapping["add"], eph(), pset.mapping["ARG0"]])


def _target_evaluate(pset: gp.PrimitiveSet, target: float):
    def evaluate(individual):
        func = gp.compile_tree(individual, pset)
        err = func(0.0) - target
        return (err * err,)

    return evaluate


def test_numeric_leaves_prefix_order_and_slim_head_then_deltas(ind_cls):
    first = _float_pset("MEMETIC_WALK_A")
    second = _float_pset("MEMETIC_WALK_B")
    head = _two_leaf_tree(ind_cls, first)
    delta = _two_leaf_tree(ind_cls, second)
    gp.assign_ephemerals(head, [1.25])
    gp.assign_ephemerals(delta, [4.5])
    slim = gp.SlimTree(head, [delta])

    locs = gp.numeric_leaves(slim)
    assert [tree[index].value for tree, index in locs] == [1.25, 4.5]
    numpy.testing.assert_allclose(gp.extract_ephemerals(slim), [1.25, 4.5])


def test_assign_ephemerals_rounds_and_clamps_window_leaves(ind_cls):
    pset = gp.make_column_pset(["value"])
    gp.add_window_primitives(pset)
    gp.add_window_ephemeral(pset, "MEMETIC_WIN", 2, 5)
    window = pset.terminals[gp.Window][0]()
    tree = ind_cls([pset.mapping["delay"], pset.mapping["value"], window])
    assert leaf_range(window) == (2.0, 5.0)

    gp.assign_ephemerals(tree, [3.6])
    assert tree[2].value == 4
    gp.assign_ephemerals(tree, [9.2])
    assert tree[2].value == 5
    gp.assign_ephemerals(tree, [0.1])
    assert tree[2].value == 2


def test_assign_ephemerals_clamps_literal_window_to_at_least_one(ind_cls):
    pset = gp.make_column_pset(["value"])
    gp.add_window_primitives(pset)
    pset.add_terminal(5, gp.Window)
    window = pset.terminals[gp.Window][0]
    tree = ind_cls([pset.mapping["delay"], pset.mapping["value"], window])

    gp.assign_ephemerals(tree, [-2.4])
    assert tree[2].value == 1


def test_assign_ephemerals_rejects_a_wrong_length(ind_cls):
    pset = _float_pset("MEMETIC_LEN")
    tree = _two_leaf_tree(ind_cls, pset)
    with pytest.raises(ValueError, match="numeric-leaf"):
        gp.assign_ephemerals(tree, [1.0, 2.0])


def test_tune_ephemerals_uses_evaluate_on_clones_and_invalidates(ind_cls):
    pset = _float_pset("MEMETIC_TUNE")
    tree = _two_leaf_tree(ind_cls, pset)
    gp.assign_ephemerals(tree, [0.0])
    tree.fitness.values = (99.0,)
    original_node = tree[1]
    compiled = gp.compile_tree(tree, pset)
    old = str(tree)
    seen = []

    def evaluate(individual):
        seen.append(individual[1] is original_node)
        return _target_evaluate(pset, 3.0)(individual)

    strategy = tools.Strategy([0.0], 0.8, offsprings=4, survivors=2)
    tools.rng.seed(7)
    tuned = gp.tune_ephemerals(tree, strategy, evaluate, n_gen=2)

    assert tuned is tree
    assert original_node.value == 0.0
    assert not any(seen)
    assert not tree.fitness.is_valid()
    assert tree[1] is not original_node
    assert gp.compile_tree(old, pset) is not compiled
    assert len(seen) == 2 * strategy.lamb


def test_tune_ephemerals_evaluate_batch_is_one_call_per_generation(ind_cls):
    pset = _float_pset("MEMETIC_BATCH")
    tree = _two_leaf_tree(ind_cls, pset)
    tree.fitness.values = (1.0,)
    batches = []

    def evaluate(_individual):
        raise AssertionError("evaluate should not run when evaluate_batch is set")

    def evaluate_batch(individuals):
        batches.append(len(individuals))
        return [(0.0,) for _ in individuals]

    strategy = tools.Strategy([0.25], 0.1, offsprings=3, survivors=2)
    tools.rng.seed(3)
    gp.tune_ephemerals(tree, strategy, evaluate, n_gen=2, evaluate_batch=evaluate_batch)

    assert batches == [strategy.lamb, strategy.lamb]


def test_tune_ephemerals_boxes_window_and_caller_bounds(ind_cls):
    pset = gp.make_column_pset(["value"])
    gp.add_window_primitives(pset)
    gp.add_window_ephemeral(pset, "MEMETIC_BOX", 2, 6)
    window = pset.terminals[gp.Window][0]()
    tree = ind_cls([pset.mapping["delay"], pset.mapping["value"], window])
    tree.fitness.values = (1.0,)
    seen = []

    def evaluate(individual):
        seen.append(gp.extract_ephemerals(individual)[0])
        return (0.0,)

    strategy = tools.Strategy([4.0], 8.0, offsprings=6, survivors=3, low=0.0, up=10.0)
    tools.rng.seed(11)
    gp.tune_ephemerals(tree, strategy, evaluate, n_gen=1)

    assert all(2.0 <= value <= 6.0 for value in seen)
    assert 2 <= tree[2].value <= 6
    assert strategy.bound_mode == "clip"


def test_tune_ephemerals_accepts_separable_strategy(ind_cls):
    pset = _float_pset("MEMETIC_SEP")
    tree = _two_leaf_tree(ind_cls, pset)
    tree.fitness.values = (1.0,)
    strategy = tools.StrategySeparable([0.25], 0.4, offsprings=4, survivors=2)
    tools.rng.seed(4)
    gp.tune_ephemerals(tree, strategy, _target_evaluate(pset, 1.0), n_gen=1)
    assert not tree.fitness.is_valid()
    assert len(gp.extract_ephemerals(tree)) == 1


def test_tune_ephemerals_no_op_without_numeric_leaves(ind_cls):
    pset = gp.PrimitiveSet("main", 1)
    tree = ind_cls([pset.mapping["ARG0"]])
    tree.fitness.values = (1.0,)
    strategy = tools.Strategy([0.0], 0.1)

    returned = gp.tune_ephemerals(tree, strategy, lambda _ind: (0.0,), n_gen=3)

    assert returned is tree
    assert tree.fitness.is_valid()


def test_tune_ephemerals_rejects_dim_mismatch(ind_cls):
    pset = _float_pset("MEMETIC_ERR")
    tree = _two_leaf_tree(ind_cls, pset)
    strategy = tools.Strategy([0.0, 0.0], 0.1)
    with pytest.raises(ValueError, match="strategy.dim"):
        gp.tune_ephemerals(tree, strategy, lambda _ind: (0.0,), n_gen=1)


def test_tune_ephemerals_rejects_bad_budget(ind_cls):
    pset = _float_pset("MEMETIC_N_GEN")
    tree = _two_leaf_tree(ind_cls, pset)
    strategy = tools.Strategy([0.0], 0.1)
    with pytest.raises(ValueError, match="n_gen"):
        gp.tune_ephemerals(tree, strategy, lambda _ind: (0.0,), n_gen=0)


def test_discard_expression_drops_raw_and_lambda_keys():
    cache = CompileCache()
    marker = object()
    cache.set(("python", 0, "add(1, 2)", ()), marker)
    cache.set(("python", 0, "lambda ARG0: add(1, 2)", ()), marker)
    cache.set(("opcode", 0, "mul(1, 2)", ()), marker)

    assert cache.discard_expression("add(1, 2)") == 2
    assert cache.get(("python", 0, "add(1, 2)", ())) is None
    assert cache.get(("opcode", 0, "mul(1, 2)", ())) is marker

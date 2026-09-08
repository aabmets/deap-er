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

import pytest
from deap_er import Fitness, creator, gp, tools
from deap_er.private.programming.compilers import clear_compile_cache, invalidate_compiled

FIT = "EVAL_CACHE_FIT"
IND = "EVAL_CACHE_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, gp.PrimitiveTree, fitness=creator.__dict__[FIT])
    yield creator.__dict__[IND]
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def _pset() -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    return pset


def _tree(ind_cls, source: str = "add(ARG0, 2)"):
    return ind_cls(gp.PrimitiveTree.from_string(source, _pset()))


def test_eval_cache_hit_does_not_repay_evaluate(ind_cls):
    calls = []

    def evaluate(individual):
        calls.append(str(individual))
        return (float(len(individual)),)

    cache = tools.EvalCache(evaluate)
    tree = _tree(ind_cls)
    first = cache.evaluate(tree)
    second = cache.evaluate(tree)

    assert first == (3.0,)
    assert second == first
    assert calls == ["add(ARG0, 2)"]


def test_eval_cache_miss_on_other_expression_or_matrix(ind_cls):
    calls = []

    def evaluate(individual):
        calls.append(str(individual))
        return (1.0,)

    left = [[0.0], [1.0]]
    cache = tools.EvalCache(evaluate, matrix=left)
    cache.evaluate(_tree(ind_cls))
    cache.evaluate(_tree(ind_cls, "add(ARG0, 3)"))
    cache.matrix = [[0.0], [1.0]]
    cache.evaluate(_tree(ind_cls))
    left.append([2.0])
    cache.matrix = left
    cache.evaluate(_tree(ind_cls))

    assert calls == ["add(ARG0, 2)", "add(ARG0, 3)", "add(ARG0, 2)", "add(ARG0, 2)"]


def test_eval_cache_batch_scores_only_misses(ind_cls):
    batches = []

    def evaluate_batch(individuals):
        batches.append([str(ind) for ind in individuals])
        return [(float(len(ind)),) for ind in individuals]

    cache = tools.EvalCache(evaluate_batch=evaluate_batch)
    known = _tree(ind_cls)
    unknown = _tree(ind_cls, "add(ARG0, 4)")
    cache.evaluate_batch([known])
    result = cache.evaluate_batch([known, unknown])

    assert result == [(3.0,), (3.0,)]
    assert batches == [["add(ARG0, 2)"], ["add(ARG0, 4)"]]


def test_eval_cache_caller_key_overrides_expression(ind_cls):
    calls = []

    def evaluate(_individual):
        calls.append(1)
        return (0.0,)

    cache = tools.EvalCache(evaluate)
    cache.evaluate(_tree(ind_cls), key="shared")
    cache.evaluate(_tree(ind_cls, "add(ARG0, 9)"), key="shared")

    assert calls == [1]


def test_invalidate_compiled_drops_eval_cache_keys(ind_cls):
    calls = []

    def evaluate(individual):
        calls.append(str(individual))
        return (1.0,)

    cache = tools.EvalCache(evaluate)
    tree = _tree(ind_cls)
    cache.evaluate(tree)
    other = _tree(ind_cls, "add(ARG0, 5)")
    cache.evaluate(other)
    assert invalidate_compiled(tree) >= 0
    cache.evaluate(tree)
    cache.evaluate(other)

    assert calls == ["add(ARG0, 2)", "add(ARG0, 5)", "add(ARG0, 2)"]


def test_promote_subtree_clears_eval_cache(ind_cls):
    calls = []

    def evaluate(individual):
        calls.append(str(individual))
        return (1.0,)

    pset = gp.PrimitiveSetTyped("main", [float, float], float)
    pset.add_primitive(operator.add, [float, float], float)
    tree = ind_cls(gp.PrimitiveTree.from_string("add(ARG0, ARG1)", pset))
    cache = tools.EvalCache(evaluate)
    cache.evaluate(tree)
    gp.promote_subtree(pset, tree)
    cache.evaluate(tree)

    assert calls == ["add(ARG0, ARG1)", "add(ARG0, ARG1)"]


def test_tune_ephemerals_drops_old_eval_cache_key(ind_cls):
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_ephemeral_constant("EVAL_CACHE_EPH", lambda: 0.25)
    eph = pset.terminals[object][-1]
    tree = ind_cls([pset.mapping["add"], eph(), pset.mapping["ARG0"]])
    gp.assign_ephemerals(tree, [0.0])
    cache_calls = []

    def evaluate(individual):
        func = gp.compile_tree(individual, pset)
        value = func(0.0)
        return (value * value,)

    def cached_evaluate(individual):
        cache_calls.append(str(individual))
        return evaluate(individual)

    snapshot = ind_cls(list(tree))
    cache = tools.EvalCache(cached_evaluate)
    cache.evaluate(tree)
    cache.evaluate(snapshot)
    strategy = tools.Strategy([0.0], 0.8, offsprings=2, survivors=1)
    tools.rng.seed(7)
    gp.tune_ephemerals(tree, strategy, evaluate, n_gen=1)
    cache.evaluate(snapshot)

    assert cache_calls == [str(snapshot), str(snapshot)]


def test_eval_cache_requires_a_callable():
    cache = tools.EvalCache()
    with pytest.raises(ValueError, match="evaluate"):
        cache.evaluate("add(ARG0, 1)")
    with pytest.raises(ValueError, match="evaluate"):
        cache.evaluate_batch(["add(ARG0, 1)"])


def test_clear_compile_cache_clears_eval_cache(ind_cls):
    cache = tools.EvalCache(lambda individual: (1.0,))
    cache.evaluate(_tree(ind_cls))
    assert len(cache) == 1
    clear_compile_cache()
    assert len(cache) == 0


def test_clear_and_invalidate_reach_every_live_eval_cache(ind_cls):
    first = tools.EvalCache(lambda individual: (1.0,))
    second = tools.EvalCache(lambda individual: (2.0,))
    tree = _tree(ind_cls)
    other = _tree(ind_cls, "add(ARG0, 7)")
    first.evaluate(tree)
    second.evaluate(tree)
    second.evaluate(other)

    assert invalidate_compiled(tree) >= 0
    assert len(first) == 0
    assert len(second) == 1
    clear_compile_cache()
    assert len(first) == 0
    assert len(second) == 0

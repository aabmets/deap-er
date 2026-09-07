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

FIT = "MEMETIC_TUNE_FIT"
IND = "MEMETIC_TUNE_IND"


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


def test_tune_ephemerals_evaluate_batch_does_not_require_evaluate(ind_cls):
    pset = _float_pset("MEMETIC_BATCH_ONLY")
    tree = _two_leaf_tree(ind_cls, pset)
    tree.fitness.values = (1.0,)
    batches = []

    def evaluate_batch(individuals):
        batches.append(len(individuals))
        return [(0.0,) for _ in individuals]

    strategy = tools.Strategy([0.25], 0.1, offsprings=3, survivors=2)
    tools.rng.seed(3)
    gp.tune_ephemerals(tree, strategy, n_gen=1, evaluate_batch=evaluate_batch)

    assert batches == [strategy.lamb]
    assert not tree.fitness.is_valid()


def test_tune_ephemerals_requires_evaluate_or_evaluate_batch(ind_cls):
    pset = _float_pset("MEMETIC_NO_EVAL")
    tree = _two_leaf_tree(ind_cls, pset)
    tree.fitness.values = (1.0,)
    strategy = tools.Strategy([0.25], 0.1)

    with pytest.raises(ValueError, match="evaluate"):
        gp.tune_ephemerals(tree, strategy, n_gen=1)


def test_tune_ephemerals_writes_back_strategy_centroid(ind_cls):
    pset = _float_pset("MEMETIC_CENTROID")
    tree = _two_leaf_tree(ind_cls, pset)
    gp.assign_ephemerals(tree, [0.0])
    tree.fitness.values = (1.0,)

    def evaluate(_individual):
        return (0.0,)

    strategy = tools.Strategy([0.0], 0.8, offsprings=4, survivors=2)
    tools.rng.seed(7)
    gp.tune_ephemerals(tree, strategy, evaluate, n_gen=2)

    numpy.testing.assert_allclose(gp.extract_ephemerals(tree), strategy.centroid)


def test_tune_ephemerals_slim_tree_invalidates_compile_cache(ind_cls):
    pset = _float_pset("MEMETIC_SLIM_TUNE")
    head = _two_leaf_tree(ind_cls, pset)
    delta = _two_leaf_tree(ind_cls, pset)
    gp.assign_ephemerals(head, [0.0])
    gp.assign_ephemerals(delta, [1.0])
    slim = gp.SlimTree(head, [delta])
    slim.fitness = type(head.fitness)()
    slim.fitness.values = (1.0,)
    compiled_head = gp.compile_tree(slim.head, pset)
    compiled_delta = gp.compile_tree(slim.deltas[0], pset)
    old_head = str(slim.head)
    old_delta = str(slim.deltas[0])

    def evaluate(individual):
        func = gp.compile_slim_tree(individual, pset)
        err = func(0.0) - 2.0
        return (err * err,)

    strategy = tools.Strategy([0.0, 1.0], 0.5, offsprings=4, survivors=2)
    tools.rng.seed(5)
    tuned = gp.tune_ephemerals(slim, strategy, evaluate, n_gen=1)

    assert tuned is slim
    assert not slim.fitness.is_valid()
    numpy.testing.assert_allclose(gp.extract_ephemerals(slim), strategy.centroid)
    assert gp.compile_tree(old_head, pset) is not compiled_head
    assert gp.compile_tree(old_delta, pset) is not compiled_delta

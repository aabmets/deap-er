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
from array import array
from typing import Any

from deap_er import Fitness, creator, gp, tools


def test_clone_individual_copies_list_genes_and_fitness():
    creator.create_type("CLONE_FIT", Fitness, weights=(1.0,))
    creator.create_type("CLONE_IND", list, fitness=creator.__dict__["CLONE_FIT"])
    try:
        original = creator.__dict__["CLONE_IND"]([1, 0, 1])
        original.fitness.values = (3.0,)
        cloned = tools.clone_individual(original)
        cloned[0] = 9
        del cloned.fitness.values
        assert list(original) == [1, 0, 1]
        assert original.fitness.values == (3.0,)
        assert cloned is not original
        assert cloned.fitness is not original.fitness
    finally:
        del creator.__dict__["CLONE_FIT"]
        del creator.__dict__["CLONE_IND"]


def test_clone_individual_copies_array_genes():
    creator.create_type("CLONE_ARR_FIT", Fitness, weights=(1.0,))
    creator.create_type(
        "CLONE_ARR_IND", array, typecode="b", fitness=creator.__dict__["CLONE_ARR_FIT"]
    )
    try:
        original = creator.__dict__["CLONE_ARR_IND"]([1, 0, 1])
        original.fitness.values = (2.0,)
        cloned = tools.clone_individual(original)
        cloned[1] = 7
        assert list(original) == [1, 0, 1]
        assert cloned.fitness.values == (2.0,)
    finally:
        del creator.__dict__["CLONE_ARR_FIT"]
        del creator.__dict__["CLONE_ARR_IND"]


def test_clone_individual_shares_gp_tree_nodes_and_splits_fitness():
    # GP nodes are immutable once created, so a genetic programming
    # toolbox can register this clone instead of deepcopy and skip
    # copying every node of every individual of every generation.
    creator.create_type("CLONE_GP_FIT", Fitness, weights=(1.0,))
    creator.create_type("CLONE_GP_IND", gp.PrimitiveTree, fitness=creator.__dict__["CLONE_GP_FIT"])
    try:
        pset = gp.make_column_pset(["value"])
        gp.add_numpy_primitives(pset)
        original = creator.__dict__["CLONE_GP_IND"](gp.gen_full(pset, 2, 3))
        original.fitness.values = (4.0,)

        cloned = tools.clone_individual(original)
        del cloned.fitness.values

        assert cloned is not original
        assert list(cloned) == list(original)
        assert all(left is right for left, right in zip(cloned, original, strict=True))
        assert cloned.fitness is not original.fitness
        assert original.fitness.values == (4.0,)
        assert not cloned.fitness.is_valid()
    finally:
        del creator.__dict__["CLONE_GP_FIT"]
        del creator.__dict__["CLONE_GP_IND"]


def test_clone_individual_falls_back_when_strategy_is_set():
    creator.create_type("CLONE_ES_FIT", Fitness, weights=(1.0,))
    creator.create_type("CLONE_ES_IND", list, fitness=creator.__dict__["CLONE_ES_FIT"])
    try:
        original = creator.__dict__["CLONE_ES_IND"]([1.0, 2.0])
        original.fitness.values = (1.0,)
        original.strategy = [0.1, 0.2]
        cloned = tools.clone_individual(original)
        cloned.strategy[0] = 9.0
        assert original.strategy == [0.1, 0.2]
        assert cloned.strategy == [9.0, 0.2]
        assert cloned is not original
    finally:
        del creator.__dict__["CLONE_ES_FIT"]
        del creator.__dict__["CLONE_ES_IND"]


class _SlotStrategy:
    __slots__ = ("strategy",)

    def __init__(self, strategy):
        self.strategy = strategy


class _SlotHistory:
    __slots__ = ("history_index",)

    def __init__(self, history_index):
        self.history_index = history_index


def test_clone_individual_falls_back_for_slots_numpy_and_tuple():
    slotted: Any = _SlotStrategy([0.1])
    cloned_slot = tools.clone_individual(slotted)
    cloned_slot.strategy[0] = 9.0
    assert slotted.strategy == [0.1]

    history: Any = _SlotHistory(3)
    cloned_history = tools.clone_individual(history)
    assert cloned_history.history_index == 3
    assert cloned_history is not history

    array_ind: Any = __import__("numpy").array([1.0, 2.0])
    cloned_array = tools.clone_individual(array_ind)
    cloned_array[0] = 9.0
    assert array_ind[0] == 1.0

    cloned_tuple: Any = (1, 2, 3)
    assert tools.clone_individual(cloned_tuple) == (1, 2, 3)

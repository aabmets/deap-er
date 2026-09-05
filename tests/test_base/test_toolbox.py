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
from copy import deepcopy
from functools import partial

from deap_er import Fitness, Toolbox, clone_individual, creator, gp


class TestToolbox:
    def test_clone_func(self):
        tb = Toolbox()
        assert isinstance(tb.clone, partial)
        assert tb.clone.func == deepcopy

    def test_map_func(self):
        tb = Toolbox()
        assert isinstance(tb.clone, partial)
        assert tb.map.func is map

    def test_registration(self):
        tb = Toolbox()
        tb.register("__test__", str, 1)
        assert hasattr(tb, "__test__")
        tb.unregister("__test__")
        assert not hasattr(tb, "__test__")

    def test_execution(self):
        tb = Toolbox()
        tb.register("__test__", str, 1)
        assert tb.__test__() == "1"

    def test_decorator(self):
        def test_deco(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return result * 3

            return wrapper

        tb = Toolbox()
        tb.register("__test__", str, 1)
        tb.decorate("__test__", test_deco)
        assert tb.__test__() == "111"


def test_clone_individual_copies_list_genes_and_fitness():
    creator.create_type("CLONE_FIT", Fitness, weights=(1.0,))
    creator.create_type("CLONE_IND", list, fitness=creator.__dict__["CLONE_FIT"])
    try:
        original = creator.__dict__["CLONE_IND"]([1, 0, 1])
        original.fitness.values = (3.0,)
        cloned = clone_individual(original)
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
    creator.create_type("CLONE_ARR_IND", array, typecode="b", fitness=creator.__dict__["CLONE_ARR_FIT"])
    try:
        original = creator.__dict__["CLONE_ARR_IND"]([1, 0, 1])
        original.fitness.values = (2.0,)
        cloned = clone_individual(original)
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

        cloned = clone_individual(original)
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
        cloned = clone_individual(original)
        cloned.strategy[0] = 9.0
        assert original.strategy == [0.1, 0.2]
        assert cloned.strategy == [9.0, 0.2]
        assert cloned is not original
    finally:
        del creator.__dict__["CLONE_ES_FIT"]
        del creator.__dict__["CLONE_ES_IND"]

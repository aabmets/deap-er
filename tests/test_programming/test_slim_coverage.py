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
import math
import operator

from deap_er import Fitness, creator, gp, tools


class _FitMin(Fitness):
    weights = (-1.0,)


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


def test_mut_slim_inflate_draws_step_when_omitted():
    tools.rng.seed(0)
    pset = _semantic_pset()
    slim = _slim_head(pset)

    (mutated,) = gp.mut_slim_inflate(slim, pset, min_depth=1, max_depth=1)

    assert mutated is slim
    assert len(slim.deltas) == 1


def test_mut_slim_respects_inflate_probability():
    tools.rng.seed(1)
    pset = _semantic_pset()
    inflated = _slim_head(pset)
    deflated = _slim_head(pset)
    gp.mut_slim_inflate(deflated, pset, min_depth=1, max_depth=1, mut_step=0.2)

    gp.mut_slim(inflated, pset, inflate_prob=1.0, min_depth=1, max_depth=1, mut_step=0.2)
    gp.mut_slim(deflated, pset, inflate_prob=0.0)

    assert len(inflated.deltas) == 1
    assert deflated.deltas == []


def test_slim_from_tree_returns_existing_instance():
    tools.rng.seed(4)
    slim = _slim_head(_semantic_pset())

    assert gp.SlimTree.from_tree(slim) is slim


def test_slim_str_and_deepcopy_copy_fitness():
    tools.rng.seed(5)
    pset = _semantic_pset()
    slim = _slim_head(pset)
    gp.mut_slim_inflate(slim, pset, min_depth=1, max_depth=1, mut_step=0.2)
    slim.fitness = _FitMin((1.0,))

    text = str(slim)
    clone = copy.deepcopy(slim)

    assert " + " in text
    assert clone is not slim
    assert clone.fitness is not slim.fitness
    assert clone.fitness.values == (1.0,)


def test_slim_deepcopy_keeps_creator_class_and_attributes():
    creator.create_type("SLIM_COPY_FIT", Fitness, weights=(-1.0,))
    creator.create_type("SLIM_COPY_IND", gp.SlimTree, fitness=creator.__dict__["SLIM_COPY_FIT"])
    try:
        ind_cls = creator.__dict__["SLIM_COPY_IND"]
        slim = ind_cls(gp.gen_grow(_semantic_pset(), 1, 2))
        slim.history_index = 7
        slim.fitness.values = (1.0,)

        clone = copy.deepcopy(slim)

        assert type(clone) is ind_cls
        assert clone.history_index == 7
        assert clone.fitness.values == (1.0,)
        assert clone.head is not slim.head
    finally:
        del creator.__dict__["SLIM_COPY_FIT"]
        del creator.__dict__["SLIM_COPY_IND"]

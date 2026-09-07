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

from deap_er import gp, tools


class _ValuesWeights:
    def __init__(self, values, weights):
        self.values = values
        self.weights = weights

    def __gt__(self, other):
        return NotImplemented


class _WValuesOnly:
    def __init__(self, values, weights, wvalues):
        self.values = values
        self.weights = weights
        self.wvalues = wvalues

    def __gt__(self, other):
        return NotImplemented


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


def test_cx_slim_donor_compares_values_and_weights():
    tools.rng.seed(2)
    pset = _semantic_pset()
    donor = _slim_head(pset)
    receiver = _slim_head(pset)
    gp.mut_slim_inflate(donor, pset, min_depth=1, max_depth=1, mut_step=0.15)
    block = str(donor.deltas[0])
    donor.fitness = _ValuesWeights((1.0,), (-1.0,))
    receiver.fitness = _ValuesWeights((10.0,), (-1.0,))

    gp.cx_slim_donor(donor, receiver, pset, best_donor=True)

    assert any(str(delta) == block for delta in receiver.deltas)


def test_cx_slim_donor_compares_wvalues_when_gt_is_unimplemented():
    tools.rng.seed(3)
    pset = _semantic_pset()
    donor = _slim_head(pset)
    receiver = _slim_head(pset)
    gp.mut_slim_inflate(donor, pset, min_depth=1, max_depth=1, mut_step=0.15)
    block = str(donor.deltas[0])
    donor.fitness = _WValuesOnly((1.0,), (-1.0,), (-1.0,))
    receiver.fitness = _WValuesOnly((10.0,), (-1.0,), (-10.0,))

    gp.cx_slim_donor(donor, receiver, pset, best_donor=True)

    assert any(str(delta) == block for delta in receiver.deltas)


def test_slim_from_tree_returns_existing_instance():
    tools.rng.seed(4)
    slim = _slim_head(_semantic_pset())

    assert gp.SlimTree.from_tree(slim) is slim


def test_slim_str_and_deepcopy_copy_fitness():
    tools.rng.seed(5)
    pset = _semantic_pset()
    slim = _slim_head(pset)
    gp.mut_slim_inflate(slim, pset, min_depth=1, max_depth=1, mut_step=0.2)
    slim.fitness = _ValuesWeights((1.0,), (-1.0,))

    text = str(slim)
    clone = copy.deepcopy(slim)

    assert " + " in text
    assert clone is not slim
    assert clone.fitness is not slim.fitness
    assert clone.fitness.values == (1.0,)

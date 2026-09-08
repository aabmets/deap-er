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


class _FitnessStub:
    """Minimal fitness stand-in for crossover donor tests."""

    def __init__(self, value: float, weight: float = -1.0) -> None:
        self.values = (value,)
        self.weights = (weight,)
        self.wvalues = (value * weight,)
        self.valid = True

    def is_valid(self) -> bool:
        return self.valid

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, _FitnessStub):
            return NotImplemented
        return self.wvalues > other.wvalues


class _MOFitnessStub:
    """Two-objective fitness stand-in for donor-selection tests."""

    def __init__(
        self,
        values: tuple[float, float],
        weights: tuple[float, float] = (-1.0, -1.0),
    ) -> None:
        self.values = values
        self.weights = weights
        self.wvalues = tuple(value * weight for value, weight in zip(values, weights, strict=True))
        self.valid = True

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, _MOFitnessStub):
            return NotImplemented
        return self.wvalues > other.wvalues


class _FitnessValidFlagStub:
    """Fitness stub exposing only a ``valid`` flag (no ``is_valid``)."""

    def __init__(self, value: float, weight: float = -1.0) -> None:
        self.values = (value,)
        self.weights = (weight,)
        self.wvalues = (value * weight,)
        self.valid = True

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, _FitnessValidFlagStub):
            return NotImplemented
        return self.wvalues > other.wvalues


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


def test_cx_slim_donor_transfers_block():
    tools.rng.seed(5)
    pset = _semantic_pset()
    slim1 = _slim_head(pset)
    slim2 = _slim_head(pset)
    gp.mut_slim_inflate(slim1, pset, min_depth=1, max_depth=1, mut_step=0.2)
    gp.mut_slim_inflate(slim1, pset, min_depth=1, max_depth=1, mut_step=0.3)
    deltas_before = len(slim1.deltas) + len(slim2.deltas)
    child1, child2 = gp.cx_slim_donor(slim1, slim2, pset, best_donor=False)
    assert len(child1.deltas) + len(child2.deltas) == deltas_before
    assert len(child1.deltas) + len(child2.deltas) == 2


def test_cx_slim_donor_best_donor():
    tools.rng.seed(6)
    pset = _semantic_pset()
    donor = _slim_head(pset)
    receiver = _slim_head(pset)
    gp.mut_slim_inflate(donor, pset, min_depth=1, max_depth=1, mut_step=0.15)
    donor_block = str(donor.deltas[0])
    donor.fitness = _FitnessStub(1.0)
    receiver.fitness = _FitnessStub(10.0)
    _, child_receiver = gp.cx_slim_donor(donor, receiver, pset, best_donor=True)
    assert any(str(delta) == donor_block for delta in child_receiver.deltas)


def test_cx_slim_donor_random_when_fitness_invalid():
    tools.rng.seed(7)
    pset = _semantic_pset()
    slim1 = _slim_head(pset)
    slim2 = _slim_head(pset)
    gp.mut_slim_inflate(slim1, pset, min_depth=1, max_depth=1, mut_step=0.2)
    child1, child2 = gp.cx_slim_donor(slim1, slim2, pset, best_donor=True)
    assert isinstance(child1, gp.SlimTree)
    assert isinstance(child2, gp.SlimTree)


def test_cx_empty_donor_deltas_noop():
    tools.rng.seed(8)
    pset = _semantic_pset()
    slim1 = _slim_head(pset)
    slim2 = _slim_head(pset)
    gp.mut_slim_inflate(slim2, pset, min_depth=1, max_depth=1, mut_step=0.2)
    slim1.fitness = _FitnessStub(1.0)
    slim2.fitness = _FitnessStub(10.0)
    deltas_before = (len(slim1.deltas), len(slim2.deltas))
    child1, child2 = gp.cx_slim_donor(slim1, slim2, pset, best_donor=True)
    assert len(child1.deltas) == deltas_before[0]
    assert len(child2.deltas) == deltas_before[1]


def test_cx_slim_donor_rejects_primitive_tree():
    tools.rng.seed(12)
    pset = _semantic_pset()
    tree1: Any = gp.gen_grow(pset, 1, 2)
    slim2 = _slim_head(pset)
    gp.mut_slim_inflate(slim2, pset, min_depth=1, max_depth=1, mut_step=0.2)
    try:
        gp.cx_slim_donor(tree1, slim2, pset, best_donor=False)
    except TypeError as err:
        assert "SlimTree" in str(err)
    else:
        raise AssertionError("expected TypeError for PrimitiveTree input")


def test_cx_slim_donor_mutates_parents_in_place():
    tools.rng.seed(13)
    pset = _semantic_pset()
    slim1 = _slim_head(pset)
    slim2 = _slim_head(pset)
    gp.mut_slim_inflate(slim1, pset, min_depth=1, max_depth=1, mut_step=0.2)
    child1, child2 = gp.cx_slim_donor(slim1, slim2, pset, best_donor=False)
    assert child1 is slim1
    assert child2 is slim2


def test_cx_slim_donor_best_donor_uses_multi_objective_fitness():
    tools.rng.seed(14)
    pset = _semantic_pset()
    worse = _slim_head(pset)
    better = _slim_head(pset)
    gp.mut_slim_inflate(worse, pset, min_depth=1, max_depth=1, mut_step=0.2)
    gp.mut_slim_inflate(better, pset, min_depth=1, max_depth=1, mut_step=0.3)
    better_block = str(better.deltas[0])
    worse.fitness = _MOFitnessStub((5.0, 10.0))
    better.fitness = _MOFitnessStub((5.0, 3.0))
    gp.cx_slim_donor(worse, better, pset, best_donor=True)
    assert len(better.deltas) == 0
    assert any(str(delta) == better_block for delta in worse.deltas)


def test_cx_slim_donor_accepts_fitness_with_valid_flag_only():
    tools.rng.seed(15)
    pset = _semantic_pset()
    donor = _slim_head(pset)
    receiver = _slim_head(pset)
    gp.mut_slim_inflate(donor, pset, min_depth=1, max_depth=1, mut_step=0.15)
    donor_block = str(donor.deltas[0])
    donor.fitness = _FitnessValidFlagStub(1.0)
    receiver.fitness = _FitnessValidFlagStub(10.0)
    _, child_receiver = gp.cx_slim_donor(donor, receiver, pset, best_donor=True)
    assert any(str(delta) == donor_block for delta in child_receiver.deltas)


def test_primitive_tree_coercion():
    tools.rng.seed(10)
    pset = _semantic_pset()
    tree1: Any = gp.gen_grow(pset, 1, 2)
    slim2 = _slim_head(pset)
    gp.mut_slim_inflate(slim2, pset, min_depth=1, max_depth=1, mut_step=0.2)
    try:
        gp.cx_slim_donor(tree1, slim2, pset, best_donor=False)
    except TypeError as err:
        assert "SlimTree" in str(err)
    else:
        raise AssertionError("expected TypeError for PrimitiveTree input")


def test_missing_primitives_raises():
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    slim = _slim_head(pset)
    try:
        gp.mut_slim_inflate(slim, pset)
    except TypeError as err:
        assert "lf" in str(err)
    else:
        raise AssertionError("expected TypeError for missing semantic primitives")

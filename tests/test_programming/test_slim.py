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

from deap_er import gp, tools


class _FitnessStub:
    """Minimal fitness stand-in for crossover donor tests."""

    def __init__(self, value: float, weight: float = -1.0) -> None:
        self.values = (value,)
        self.weights = (weight,)
        self.valid = True

    def is_valid(self) -> bool:
        return self.valid


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
    (mutated,) = gp.mut_slim_deflate(slim, pset)
    assert len(mutated.deltas) == 1
    assert len(mutated) < size_before


def test_slim_deflate_noop_without_deltas():
    tools.rng.seed(2)
    pset = _semantic_pset()
    slim = _slim_head(pset)
    (mutated,) = gp.mut_slim_deflate(slim, pset)
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
    assert (len(child1.deltas), len(child2.deltas)) == deltas_before


def test_deflate_never_removes_head():
    tools.rng.seed(9)
    pset = _semantic_pset()
    slim = _slim_head(pset)
    head_snapshot = list(slim.head)
    for step in range(3):
        gp.mut_slim_inflate(slim, pset, min_depth=1, max_depth=1, mut_step=0.1 * (step + 1))
        gp.mut_slim_deflate(slim, pset)
    assert [node.name for node in slim.head] == [node.name for node in head_snapshot]


def test_primitive_tree_coercion():
    tools.rng.seed(10)
    pset = _semantic_pset()
    tree1 = gp.gen_grow(pset, 1, 2)
    slim2 = _slim_head(pset)
    gp.mut_slim_inflate(slim2, pset, min_depth=1, max_depth=1, mut_step=0.2)
    child1, child2 = gp.cx_slim_donor(tree1, slim2, pset, best_donor=False)
    assert isinstance(child1, gp.SlimTree)
    assert isinstance(child2, gp.SlimTree)


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

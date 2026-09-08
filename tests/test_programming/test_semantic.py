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

from deap_er import gp


def lf(x):
    return 1 / (1 + math.exp(-x))


def test_semantic_crossover():
    pset = gp.PrimitiveSet("main", 2)
    pset.add_primitive(operator.sub, 2)
    pset.add_terminal(3)
    pset.add_primitive(lf, 1, name="lf")
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.mul, 2)
    ind1 = gp.PrimitiveTree.from_string("add(ARG0, ARG1)", pset)
    ind2 = gp.PrimitiveTree.from_string("sub(3, ARG0)", pset)
    saved1 = [node.name for node in ind1]
    saved2 = [node.name for node in ind2]
    new_ind1, new_ind2 = gp.cx_semantic(list(ind1), list(ind2), pset, max_depth=2)
    assert [node.name for node in new_ind1[2 : 2 + len(saved1)]] == saved1
    assert [node.name for node in new_ind2[2 : 2 + len(saved2)]] == saved2


def test_semantic_crossover_second_child_uses_original_first_parent():
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.mul, 2)
    pset.add_primitive(operator.sub, 2)
    pset.add_primitive(lf, 1, name="lf")
    pset.add_terminal(1.0)
    # Distinct fixed parents: random gen_grow trees often share a name
    # sequence, which made both offspring identical and this check flake.
    ind1 = gp.PrimitiveTree.from_string("add(ARG0, ARG0)", pset)
    ind2 = gp.PrimitiveTree.from_string("mul(1.0, 1.0)", pset)
    original_first = [node.name for node in ind1]
    offspring1, offspring2 = gp.cx_semantic(list(ind1), list(ind2), pset, min_depth=1, max_depth=1)
    first_offspring_names = [node.name for node in offspring1]
    second_names = [node.name for node in offspring2]
    assert second_names[-len(original_first) :] == original_first
    assert len(offspring1) == len(offspring2)
    embedded = any(
        second_names[i : i + len(first_offspring_names)] == first_offspring_names
        for i in range(len(second_names))
    )
    assert not embedded


def test_semantic_mutation():
    pset = gp.PrimitiveSet("main", 2)
    pset.add_primitive(operator.sub, 2)
    pset.add_terminal(3)
    pset.add_primitive(lf, 1, name="lf")
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.mul, 2)
    individual = gp.PrimitiveTree.from_string("add(ARG0, ARG1)", pset)
    saved = [node.name for node in individual]
    mutated = gp.mut_semantic(list(individual), pset, max_depth=2)
    assert [node.name for node in mutated[0][1 : 1 + len(saved)]] == saved

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

from deap_er.gp import cx_semantic, mut_semantic
from deap_er.gp.generators import gen_grow
from deap_er.gp.primitives import PrimitiveSet


def lf(x):
    return 1 / (1 + math.exp(-x))


def test_semantic_crossover():
    pset = PrimitiveSet("main", 2)
    pset.add_primitive(operator.sub, 2)
    pset.add_terminal(3)
    pset.add_primitive(lf, 1, name="lf")
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.mul, 2)
    ind1 = gen_grow(pset, 1, 3)
    ind2 = gen_grow(pset, 1, 3)
    new_ind1, new_ind2 = cx_semantic(ind1, ind2, pset, max_depth=2)
    ctr = sum([n.name == ind1[i].name for i, n in enumerate(new_ind1)])
    assert ctr == len(ind1)
    ctr = sum([n.name == ind2[i].name for i, n in enumerate(new_ind2)])
    assert ctr == len(ind2)


def test_semantic_crossover_second_child_uses_original_first_parent():
    pset = PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.mul, 2)
    pset.add_primitive(operator.sub, 2)
    pset.add_primitive(lf, 1, name="lf")
    pset.add_terminal(1.0)
    ind1 = gen_grow(pset, 1, 2)
    ind2 = gen_grow(pset, 1, 2)
    offspring1, offspring2 = cx_semantic(list(ind1), list(ind2), pset, min_depth=1, max_depth=1)
    first_offspring_names = [node.name for node in offspring1]
    second_names = [node.name for node in offspring2]
    embedded = any(
        second_names[i : i + len(first_offspring_names)] == first_offspring_names
        for i in range(len(second_names))
    )
    assert not embedded


def test_semantic_mutation():
    pset = PrimitiveSet("main", 2)
    pset.add_primitive(operator.sub, 2)
    pset.add_terminal(3)
    pset.add_primitive(lf, 1, name="lf")
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.mul, 2)
    individual = gen_grow(pset, 1, 3)
    mutated = mut_semantic(individual, pset, max_depth=2)
    ctr = sum([m.name == individual[i].name for i, m in enumerate(mutated[0])])
    assert ctr == len(individual)

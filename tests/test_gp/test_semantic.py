#
#   MIT License
#   
#   Copyright (c) 2022, Mattias Aabmets
#   
#   The contents of this file are subject to the terms and conditions defined in the License.
#   You may not use, modify, or distribute this file except in compliance with the License.
#   
#   SPDX-License-Identifier: MIT
#
from deap_er.gp.primitives import PrimitiveSet
from deap_er.gp.generators import gen_grow
from deap_er.gp import cx_semantic, mut_semantic
import operator
import math


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

# ====================================================================================== #
#                                                                                        #
#   MIT License                                                                          #
#                                                                                        #
#   Copyright (c) 2022 - Mattias Aabmets, The DEAP Team and Other Contributors           #
#                                                                                        #
#   Permission is hereby granted, free of charge, to any person obtaining a copy         #
#   of this software and associated documentation files (the "Software"), to deal        #
#   in the Software without restriction, including without limitation the rights         #
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell            #
#   copies of the Software, and to permit persons to whom the Software is                #
#   furnished to do so, subject to the following conditions:                             #
#                                                                                        #
#   The above copyright notice and this permission notice shall be included in all       #
#   copies or substantial portions of the Software.                                      #
#                                                                                        #
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR           #
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,             #
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE          #
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER               #
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,        #
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE        #
#   SOFTWARE.                                                                            #
#                                                                                        #
# ====================================================================================== #
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
    ctr = sum([m.name == individual[i].name for i, m in enumerate(mutated)])
    assert ctr == len(individual)

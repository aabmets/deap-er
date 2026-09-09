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
import operator

import pytest
from deap_er import Fitness, Toolbox, creator, gp, tools

REG_FIT = "REG_GP_FIT"
REG_IND = "REG_GP_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(REG_FIT, Fitness, weights=(-1.0,))
    creator.create_type(REG_IND, gp.PrimitiveTree, fitness=creator.__dict__[REG_FIT])
    yield creator.__dict__[REG_IND]
    del creator.__dict__[REG_FIT]
    del creator.__dict__[REG_IND]


def _pset():
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_primitive(operator.neg, 1)
    pset.add_terminal(1.0)
    return pset


def test_register_gp_wires_clone_compile_and_variation(ind_cls):
    toolbox = Toolbox()
    gp.register_gp(toolbox, _pset(), individual=ind_cls, height_limit=6, max_depth=2)

    assert toolbox.clone.func is tools.clone_individual
    population = toolbox.population(size=4)
    assert len(population) == 4
    assert all(isinstance(ind, gp.PrimitiveTree) for ind in population)
    compiled = toolbox.compile(expr=population[0])
    assert compiled(3.0) == compiled(3.0)
    mates = toolbox.mate(toolbox.clone(population[0]), toolbox.clone(population[1]))
    (mutant,) = toolbox.mutate(toolbox.clone(population[0]))
    assert len(mates) == 2
    assert isinstance(mutant, gp.PrimitiveTree)
    selected = toolbox.select(population, len(population))
    assert len(selected) == 4


def test_register_gp_skips_init_and_select_when_asked():
    toolbox = Toolbox()
    gp.register_gp(toolbox, _pset(), select=False, height_limit=None)

    assert not hasattr(toolbox, "individual")
    assert not hasattr(toolbox, "select")
    assert hasattr(toolbox, "mate")
    assert hasattr(toolbox, "mutate")


def test_register_gp_accepts_a_custom_select(ind_cls):
    toolbox = Toolbox()
    gp.register_gp(toolbox, _pset(), individual=ind_cls, select=tools.sel_random)
    population = toolbox.population(size=3)
    assert len(toolbox.select(population, 2)) == 2

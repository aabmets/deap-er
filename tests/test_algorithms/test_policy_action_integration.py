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

import numpy
import pytest
from deap_er import Fitness, Toolbox, creator, gp, tools

POLICY_INT_FIT = "POLICY_INT_FIT"
POLICY_INT_IND = "POLICY_INT_IND"
POLICY_INT_GP_FIT = "POLICY_INT_GP_FIT"
POLICY_INT_GP_IND = "POLICY_INT_GP_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(POLICY_INT_FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(POLICY_INT_IND, list, fitness=creator.__dict__[POLICY_INT_FIT])
    yield creator.__dict__[POLICY_INT_IND]
    del creator.__dict__[POLICY_INT_FIT]
    del creator.__dict__[POLICY_INT_IND]


@pytest.fixture
def gp_ind_cls():
    creator.create_type(POLICY_INT_GP_FIT, Fitness, weights=(-1.0,))
    creator.create_type(
        POLICY_INT_GP_IND,
        gp.PrimitiveTree,
        fitness=creator.__dict__[POLICY_INT_GP_FIT],
    )
    yield creator.__dict__[POLICY_INT_GP_IND]
    del creator.__dict__[POLICY_INT_GP_FIT]
    del creator.__dict__[POLICY_INT_GP_IND]


def test_next_lexicase_cases_integration(ind_cls):
    elites = [ind_cls([0]), ind_cls([1])]
    for elite in elites:
        elite.fitness.values = (0.0, 1.0)
    exam = tools.CaseExam.from_cases([0, 1], 2)
    result = tools.apply_policy_action(
        "next_lexicase_cases",
        exams=[exam],
        elites=elites,
        mut_prob=0.0,
    )
    assert result.applied is True
    assert result.value == [0, 1]


def test_evaluate_invalid_integration(ind_cls):
    toolbox = Toolbox()

    def evaluate(individual):
        return (float(individual[0]), 0.0)

    toolbox.register("evaluate", evaluate)
    population = [ind_cls([3.0])]
    result = tools.apply_policy_action(
        "evaluate_invalid",
        toolbox=toolbox,
        individuals=population,
    )
    assert result.applied is True
    assert result.value == 1
    assert population[0].fitness.values == (3.0, 0.0)


def test_promote_subtree_integration():
    pset = gp.PrimitiveSetTyped("main", [float, float], float)
    pset.add_primitive(operator.add, [float, float], float)
    tree = gp.PrimitiveTree.from_string("add(ARG0, ARG1)", pset)
    result = tools.apply_policy_action(
        "promote_subtree",
        prim_set=pset,
        expr=tree,
    )
    assert result.applied is True
    assert result.value == "promo0"
    assert "promo0" in gp.promoted_names(pset)


def _memetic_pset(name: str) -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_ephemeral_constant(name, lambda: 0.25)
    return pset


def _memetic_tree(gp_ind_cls, pset: gp.PrimitiveSet):
    eph = pset.terminals[object][-1]
    return gp_ind_cls([pset.mapping["add"], eph(), pset.mapping["ARG0"]])


def test_tune_ephemerals_integration(gp_ind_cls):
    pset = _memetic_pset("POLICY_INT_MEMETIC")
    tree = _memetic_tree(gp_ind_cls, pset)
    gp.assign_ephemerals(tree, [0.0])
    tree.fitness.values = (99.0,)

    def evaluate(individual):
        func = gp.compile_tree(individual, pset)
        err = func(0.0) - 3.0
        return (err * err,)

    strategy = tools.Strategy([0.0], 0.8, offsprings=4, survivors=2)
    tools.rng.seed(7)
    result = tools.apply_policy_action(
        "tune_ephemerals",
        individual=tree,
        strategy=strategy,
        evaluate=evaluate,
        n_gen=1,
    )
    assert result.applied is True
    assert result.value is tree
    assert not tree.fitness.is_valid()


def test_interpret_tapes_integration():
    pset = gp.make_column_pset(["value"])
    gp.add_numpy_primitives(pset)
    pset.add_terminal(2.5, gp.Array, "two_half")
    tape = gp.lower_tree(gp.PrimitiveTree([pset.mapping["two_half"]]), pset)
    matrix = numpy.array([[1.0], [2.0], [3.0]])
    result = tools.apply_policy_action(
        "interpret_tapes",
        tapes=[tape],
        matrix=matrix,
    )
    assert result.applied is True
    numpy.testing.assert_allclose(result.value[0], 2.5)


def test_step_islands_integration():
    fit_name = "POLICY_INT_ISLAND_FIT"
    ind_name = "POLICY_INT_ISLAND_IND"
    creator.create_type(fit_name, Fitness, weights=(1.0,))
    creator.create_type(ind_name, list, fitness=creator.__dict__[fit_name])
    ind_cls = creator.__dict__[ind_name]

    def vary(population):
        return list(population) + [ind_cls([10.0]), ind_cls([11.0])]

    toolbox = Toolbox()
    toolbox.register("evaluate", lambda ind: (float(ind[0]),))
    toolbox.register("vary", vary)
    toolbox.register("select", tools.sel_best)
    population = [ind_cls([0.0]), ind_cls([1.0])]
    try:
        result = tools.apply_policy_action(
            "step_islands",
            demes=[(toolbox, population)],
        )
    finally:
        del creator.__dict__[fit_name]
        del creator.__dict__[ind_name]

    assert result.applied is True
    assert result.rejected is False
    assert sorted(ind[0] for ind in population) == [10.0, 11.0]

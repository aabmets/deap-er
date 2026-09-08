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

POLICY_GUARD_STRESS_FIT = "POLICY_GUARD_STRESS_FIT"
POLICY_GUARD_STRESS_IND = "POLICY_GUARD_STRESS_IND"
POLICY_GUARD_STRESS_GP_FIT = "POLICY_GUARD_STRESS_GP_FIT"
POLICY_GUARD_STRESS_GP_IND = "POLICY_GUARD_STRESS_GP_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(POLICY_GUARD_STRESS_FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(
        POLICY_GUARD_STRESS_IND,
        list,
        fitness=creator.__dict__[POLICY_GUARD_STRESS_FIT],
    )
    yield creator.__dict__[POLICY_GUARD_STRESS_IND]
    del creator.__dict__[POLICY_GUARD_STRESS_FIT]
    del creator.__dict__[POLICY_GUARD_STRESS_IND]


@pytest.fixture
def gp_ind_cls():
    creator.create_type(POLICY_GUARD_STRESS_GP_FIT, Fitness, weights=(-1.0,))
    creator.create_type(
        POLICY_GUARD_STRESS_GP_IND,
        gp.PrimitiveTree,
        fitness=creator.__dict__[POLICY_GUARD_STRESS_GP_FIT],
    )
    yield creator.__dict__[POLICY_GUARD_STRESS_GP_IND]
    del creator.__dict__[POLICY_GUARD_STRESS_GP_FIT]
    del creator.__dict__[POLICY_GUARD_STRESS_GP_IND]


def _elites(ind_cls):
    elites = [ind_cls([0]), ind_cls([1])]
    for elite in elites:
        elite.fitness.values = (0.0, 1.0)
    return elites


def test_n_evals_budget_rejects_tune_without_crash(gp_ind_cls):
    guard = tools.PolicyActionGuard(n_evals=4, max_tune_gen=5)
    pset = gp.PrimitiveSet("budget_guard_main", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_ephemeral_constant("budget_guard_eph", lambda: 0.25)
    eph = pset.terminals[object][-1]
    tree = gp_ind_cls([pset.mapping["add"], eph(), pset.mapping["ARG0"]])
    strategy = tools.Strategy([0.0], 0.8, offsprings=4, survivors=2)
    result = tools.apply_policy_action(
        "tune_ephemerals",
        individual=tree,
        strategy=strategy,
        evaluate=lambda _: (1.0,),
        n_gen=2,
        guard=guard,
    )
    assert result.rejected is True


def test_n_evals_budget_allows_cheaper_action(ind_cls):
    guard = tools.PolicyActionGuard(n_evals=1)
    toolbox = Toolbox()
    toolbox.register("evaluate", lambda ind: (float(ind[0]), 0.0))
    population = [ind_cls([0])]
    result = tools.apply_policy_action(
        "evaluate_invalid",
        toolbox=toolbox,
        individuals=population,
        guard=guard,
    )
    assert result.applied is True
    assert guard.nevals_used == 1
    second = tools.apply_policy_action(
        "evaluate_invalid",
        toolbox=toolbox,
        individuals=[ind_cls([1]), ind_cls([2])],
        guard=guard,
    )
    assert second.rejected is True


def test_random_policy_runs_many_generations_without_cache_melt(ind_cls, gp_ind_cls):
    guard = tools.PolicyActionGuard(
        max_promotes_per_gen=1,
        promote_cooldown=5,
        max_tune_gen=1,
        min_exam_size=1,
        n_evals=10_000,
    )
    elites = _elites(ind_cls)
    exam = tools.CaseExam.from_cases([0, 1], 2)
    pset = gp.PrimitiveSet("rand_policy_main", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_ephemeral_constant("rand_policy_eph", lambda: 0.25)
    eph = pset.terminals[object][-1]
    tree = gp_ind_cls([pset.mapping["add"], eph(), pset.mapping["ARG0"]])
    gp.assign_ephemerals(tree, [0.0])
    strategy = tools.Strategy([0.0], 0.8, offsprings=2, survivors=1)
    toolbox = Toolbox()
    toolbox.register("evaluate", lambda ind: (float(ind[0]),))
    toolbox.register("vary", lambda pop: list(pop))
    toolbox.register("select", tools.sel_best)
    matrix = numpy.array([[1.0], [2.0]])
    tape = gp.lower_tree(gp.PrimitiveTree([pset.mapping["ARG0"]]), pset)
    actions = sorted(tools.SUPPORTED_POLICY_ACTIONS)
    tools.rng.seed(19)
    for gen in range(500):
        guard.begin_generation(gen)
        action = tools.rng.choice(actions)
        kwargs: dict[str, object] = {"guard": guard}
        if action == "next_lexicase_cases":
            kwargs.update(exams=[exam], elites=elites, mut_prob=0.0)
        elif action == "tune_ephemerals":
            kwargs.update(
                individual=tree,
                strategy=strategy,
                evaluate=lambda _: (1.0,),
                n_gen=1,
            )
        elif action == "promote_subtree":
            kwargs.update(prim_set=pset, expr=tree)
        elif action == "evaluate_invalid":
            kwargs.update(toolbox=toolbox, individuals=elites)
        elif action == "interpret_tapes":
            kwargs.update(tapes=[tape], matrix=matrix)
        elif action == "step_islands":
            kwargs.update(demes=[(toolbox, elites)])
        result = tools.apply_policy_action(action, **kwargs)
        assert isinstance(result, tools.PolicyActionResult)

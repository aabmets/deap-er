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
from unittest import mock

import numpy
import pytest
from deap_er import Fitness, Toolbox, creator, gp, tools

POLICY_GUARD_FIT = "POLICY_GUARD_FIT"
POLICY_GUARD_IND = "POLICY_GUARD_IND"
POLICY_GUARD_GP_FIT = "POLICY_GUARD_GP_FIT"
POLICY_GUARD_GP_IND = "POLICY_GUARD_GP_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(POLICY_GUARD_FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(POLICY_GUARD_IND, list, fitness=creator.__dict__[POLICY_GUARD_FIT])
    yield creator.__dict__[POLICY_GUARD_IND]
    del creator.__dict__[POLICY_GUARD_FIT]
    del creator.__dict__[POLICY_GUARD_IND]


@pytest.fixture
def gp_ind_cls():
    creator.create_type(POLICY_GUARD_GP_FIT, Fitness, weights=(-1.0,))
    creator.create_type(
        POLICY_GUARD_GP_IND,
        gp.PrimitiveTree,
        fitness=creator.__dict__[POLICY_GUARD_GP_FIT],
    )
    yield creator.__dict__[POLICY_GUARD_GP_IND]
    del creator.__dict__[POLICY_GUARD_GP_FIT]
    del creator.__dict__[POLICY_GUARD_GP_IND]


def _elites(ind_cls):
    elites = [ind_cls([0]), ind_cls([1])]
    for elite in elites:
        elite.fitness.values = (0.0, 1.0)
    return elites


def test_max_promotes_per_generation_rejects_without_crash(ind_cls):
    guard = tools.PolicyActionGuard(max_promotes_per_gen=1, promote_cooldown=0)
    guard.begin_generation(0)
    pset = gp.PrimitiveSetTyped("guard_main", [float, float], float)
    pset.add_primitive(operator.add, [float, float], float)
    tree = gp.PrimitiveTree.from_string("add(ARG0, ARG1)", pset)
    first = tools.apply_policy_action(
        "promote_subtree",
        prim_set=pset,
        expr=tree,
        guard=guard,
    )
    second = tools.apply_policy_action(
        "promote_subtree",
        prim_set=pset,
        expr=tree,
        guard=guard,
    )
    assert first.applied is True
    assert first.rejected is False
    assert second.applied is False
    assert second.rejected is True


def test_promote_cooldown_rejects_without_crash(ind_cls):
    guard = tools.PolicyActionGuard(max_promotes_per_gen=5, promote_cooldown=3)
    pset = gp.PrimitiveSetTyped("cool_main", [float, float], float)
    pset.add_primitive(operator.add, [float, float], float)
    tree = gp.PrimitiveTree.from_string("add(ARG0, ARG1)", pset)
    guard.begin_generation(0)
    assert tools.apply_policy_action(
        "promote_subtree",
        prim_set=pset,
        expr=tree,
        guard=guard,
    ).applied
    guard.begin_generation(1)
    rejected = tools.apply_policy_action(
        "promote_subtree",
        prim_set=pset,
        expr=tree,
        guard=guard,
    )
    assert rejected.rejected is True
    guard.begin_generation(3)
    allowed = tools.apply_policy_action(
        "promote_subtree",
        prim_set=pset,
        expr=tree,
        guard=guard,
    )
    assert allowed.applied is True


def test_max_tune_generations_rejects_without_crash(gp_ind_cls):
    guard = tools.PolicyActionGuard(max_tune_gen=2)
    pset = gp.PrimitiveSet("tune_guard_main", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_ephemeral_constant("tune_guard_eph", lambda: 0.25)
    eph = pset.terminals[object][-1]
    tree = gp_ind_cls([pset.mapping["add"], eph(), pset.mapping["ARG0"]])
    strategy = tools.Strategy([0.0], 0.8, offsprings=4, survivors=2)
    rejected = tools.apply_policy_action(
        "tune_ephemerals",
        individual=tree,
        strategy=strategy,
        evaluate=lambda _: (1.0,),
        n_gen=5,
        guard=guard,
    )
    assert rejected.rejected is True
    assert rejected.applied is False


@mock.patch("deap_er.private.algorithms.policy_action.tune_ephemerals", return_value="tuned")
def test_allowed_tune_still_dispatches(mock_tune, gp_ind_cls):
    guard = tools.PolicyActionGuard(max_tune_gen=5)
    result = tools.apply_policy_action(
        "tune_ephemerals",
        individual="tree",
        strategy="strategy",
        evaluate=lambda _: (1.0,),
        n_gen=3,
        guard=guard,
    )
    mock_tune.assert_called_once()
    assert result.applied is True
    assert result.value == "tuned"


def test_minimum_exam_size_rejects_small_floor(ind_cls):
    guard = tools.PolicyActionGuard(min_exam_size=2)
    elites = _elites(ind_cls)
    exam = tools.CaseExam.from_cases([0], 2)
    result = tools.apply_policy_action(
        "next_lexicase_cases",
        exams=[exam],
        elites=elites,
        min_cases=1,
        mut_prob=0.0,
        guard=guard,
    )
    assert result.rejected is True


def test_minimum_exam_size_rejects_small_exam(ind_cls):
    guard = tools.PolicyActionGuard(min_exam_size=2)
    elites = _elites(ind_cls)
    exam = tools.CaseExam.from_cases([0], 2)
    result = tools.apply_policy_action(
        "next_lexicase_cases",
        exams=[exam],
        elites=elites,
        mut_prob=0.0,
        guard=guard,
    )
    assert result.rejected is True


@mock.patch(
    "deap_er.private.algorithms.policy_action.next_lexicase_cases",
    return_value=[0, 1],
)
def test_allowed_exam_still_dispatches(mock_next, ind_cls):
    guard = tools.PolicyActionGuard(min_exam_size=1)
    elites = _elites(ind_cls)
    exam = tools.CaseExam.from_cases([0, 1], 2)
    result = tools.apply_policy_action(
        "next_lexicase_cases",
        exams=[exam],
        elites=elites,
        mut_prob=0.0,
        guard=guard,
    )
    mock_next.assert_called_once()
    assert result.applied is True


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


def test_guard_rejection_surfaces_last_action_rejected_observation(ind_cls):
    """Guard caps feed P11 ``last_action_rejected`` via ``rejected=True``."""
    guard = tools.PolicyActionGuard(max_tune_gen=1)
    pset = gp.PrimitiveSet("obs_guard_main", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_ephemeral_constant("obs_guard_eph", lambda: 0.25)
    eph = pset.terminals[object][-1]
    tree = gp.PrimitiveTree([pset.mapping["add"], eph(), pset.mapping["ARG0"]])
    strategy = tools.Strategy([0.0], 0.8, offsprings=2, survivors=1)
    result = tools.apply_policy_action(
        "tune_ephemerals",
        individual=tree,
        strategy=strategy,
        evaluate=lambda _: (1.0,),
        n_gen=3,
        guard=guard,
    )
    assert result.applied is False
    assert result.rejected is True
    assert result.value is None


def test_max_promotes_per_gen_zero_blocks_apply_without_crash(ind_cls):
    """``max_promotes_per_gen=0`` is a permanent promote ban for this generation."""
    guard = tools.PolicyActionGuard(max_promotes_per_gen=0)
    guard.begin_generation(0)
    pset = gp.PrimitiveSetTyped("zero_promo_main", [float, float], float)
    pset.add_primitive(operator.add, [float, float], float)
    tree = gp.PrimitiveTree.from_string("add(ARG0, ARG1)", pset)
    result = tools.apply_policy_action(
        "promote_subtree",
        prim_set=pset,
        expr=tree,
        guard=guard,
    )
    assert result.applied is False
    assert result.rejected is True


def test_policy_action_guard_rejects_negative_caps():
    with pytest.raises(ValueError, match="max_promotes_per_gen"):
        tools.PolicyActionGuard(max_promotes_per_gen=-1)


def test_policy_action_result_docstring_mentions_guard_rejection():
    rejected_doc = tools.PolicyActionResult.__doc__ or ""
    assert "guard" in rejected_doc.lower()


def test_guard_policy_action_matches_apply_rejection(ind_cls):
    guard = tools.PolicyActionGuard(max_promotes_per_gen=0)
    assert tools.guard_policy_action("promote_subtree", guard) is False


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

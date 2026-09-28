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

POLICY_FIT = "POLICY_FIT"
POLICY_IND = "POLICY_IND"
POLICY_GP_FIT = "POLICY_GP_FIT"
POLICY_GP_IND = "POLICY_GP_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(POLICY_FIT, Fitness, weights=(-1.0, -1.0))
    creator.create_type(POLICY_IND, list, fitness=creator.__dict__[POLICY_FIT])
    yield creator.__dict__[POLICY_IND]
    del creator.__dict__[POLICY_FIT]
    del creator.__dict__[POLICY_IND]


@pytest.fixture
def gp_ind_cls():
    creator.create_type(POLICY_GP_FIT, Fitness, weights=(-1.0,))
    creator.create_type(POLICY_GP_IND, gp.PrimitiveTree, fitness=creator.__dict__[POLICY_GP_FIT])
    yield creator.__dict__[POLICY_GP_IND]
    del creator.__dict__[POLICY_GP_FIT]
    del creator.__dict__[POLICY_GP_IND]


def _promote_pset():
    pset = gp.PrimitiveSetTyped("main", [float, float], float)
    pset.add_primitive(operator.add, [float, float], float)
    return pset


def test_supported_actions_match_skip_and_dispatch_tokens():
    assert tools.POLICY_ACTION_SKIP_TUNE in tools.SKIP_POLICY_ACTIONS
    assert tools.POLICY_ACTION_SKIP_PROMOTE in tools.SKIP_POLICY_ACTIONS
    assert tools.SKIP_POLICY_ACTIONS <= tools.SUPPORTED_POLICY_ACTIONS


def test_unknown_action_is_rejected():
    result = tools.apply_policy_action("load_column")
    assert result.applied is False
    assert result.rejected is True
    assert result.value is None


def test_skip_actions_noop_without_calling_underlying():
    pset = _promote_pset()
    tree = gp.PrimitiveTree.from_string("add(ARG0, ARG1)", pset)
    tune_result = tools.apply_policy_action(tools.POLICY_ACTION_SKIP_TUNE)
    promote_result = tools.apply_policy_action(
        tools.POLICY_ACTION_SKIP_PROMOTE, prim_set=pset, expr=tree
    )
    assert tune_result == tools.PolicyActionResult(applied=False, rejected=False)
    assert promote_result == tools.PolicyActionResult(applied=False, rejected=False)
    assert gp.promoted_names(pset) == []


@pytest.mark.parametrize(
    ("action", "kwargs"),
    [
        ("next_lexicase_cases", {"elites": []}),
        ("tune_ephemerals", {"individual": "tree"}),
        ("promote_subtree", {"expr": "tree"}),
        ("evaluate_invalid", {"individuals": []}),
        ("interpret_tapes", {"tapes": []}),
        ("step_islands", {}),
    ],
)
def test_missing_required_kwargs_are_rejected(action, kwargs):
    result = tools.apply_policy_action(action, **kwargs)
    assert result.rejected is True
    assert result.applied is False


def test_underlying_callable_errors_propagate(ind_cls):
    """Schema rejection is only for unknown tokens and missing kwargs."""
    exam = tools.CaseExam.from_cases([0], 1)
    with pytest.raises(ValueError, match="individuals must be non-empty"):
        tools.apply_policy_action(
            "next_lexicase_cases",
            exams=[exam],
            elites=[],
            mut_prob=0.0,
        )


def test_tune_ephemerals_forwards_evaluate_and_n_gen(gp_ind_cls):
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)
    pset.add_ephemeral_constant("POLICY_DISPATCH_EPH", lambda: 0.25)
    eph = pset.terminals[object][-1]
    tree = gp_ind_cls([pset.mapping["add"], eph(), pset.mapping["ARG0"]])
    gp.assign_ephemerals(tree, [0.0])
    calls = []

    def evaluate(individual):
        calls.append(individual)
        return ((gp.compile_tree(individual, pset)(0.0) - 3.0) ** 2,)

    tools.rng.seed(7)
    result = tools.apply_policy_action(
        "tune_ephemerals",
        individual=tree,
        strategy=tools.Strategy([0.0], 0.8, offsprings=4, survivors=2),
        evaluate=evaluate,
        n_gen=2,
    )
    assert result.applied is True
    assert result.value is tree
    assert len(calls) == 2 * 4


def test_tune_ephemerals_rejects_without_evaluate_callable():
    result = tools.apply_policy_action(
        "tune_ephemerals",
        individual="tree",
        strategy="strategy",
    )
    assert result.rejected is True


def test_promote_subtree_forwards_prefix():
    pset = _promote_pset()
    tree = gp.PrimitiveTree.from_string("add(ARG0, ARG1)", pset)
    result = tools.apply_policy_action(
        "promote_subtree",
        prim_set=pset,
        expr=tree,
        index=0,
        prefix="lib",
    )
    assert result.applied is True
    assert result.value == "lib0"
    assert gp.promoted_names(pset) == ["lib0"]


def test_step_islands_forwards_migrate_and_eval_keys(ind_cls):
    toolbox = Toolbox()
    toolbox.register("evaluate", lambda individual: (float(individual[0]), 0.0))
    toolbox.register("vary", list)
    toolbox.register("select", tools.sel_best)
    first = [ind_cls([0.0])]
    second = [ind_cls([1.0])]
    immigrant = second[0]

    def migrate(populations):
        populations[0][:] = populations[1][:1]

    result = tools.apply_policy_action(
        "step_islands",
        demes=[(toolbox, first), (toolbox, second)],
        migrate=migrate,
        eval_keys=("a", "b"),
    )
    assert result == tools.PolicyActionResult(applied=True, rejected=False)
    assert first == [immigrant]
    assert first[0] is immigrant
    assert not immigrant.fitness.is_valid()

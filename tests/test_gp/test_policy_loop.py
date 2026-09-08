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
import inspect
import operator
from unittest import mock

import numpy
import pytest
from deap_er import Fitness, creator, gp, tools
from deap_er.private.programming.policy_linear import (
    LinearPolicyProgram,
    PolicyDecisionRule,
    linear_policy_decide,
)
from deap_er.private.programming.policy_loop import (
    POLICY_LOOP_ACTIONS,
    policy_action_index,
    step_policy_loop,
)

POLICY_FIT = "P19_POLICY_FIT"
POLICY_IND = "P19_POLICY_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(POLICY_FIT, Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
    creator.create_type(POLICY_IND, list, fitness=creator.__dict__[POLICY_FIT])
    yield creator.__dict__[POLICY_IND]
    del creator.__dict__[POLICY_FIT]
    del creator.__dict__[POLICY_IND]


def _elite(ind_cls, values):
    individual = ind_cls([0])
    individual.fitness.values = values
    return individual


def _pool_and_elites(ind_cls):
    elites = [
        _elite(ind_cls, (0.0, 1.0, 0.0, 1.0)),
        _elite(ind_cls, (0.0, 0.0, 1.0, 1.0)),
    ]
    train = tools.CaseExam.from_cases([1, 3], 4)
    held = tools.CaseExam.from_cases([2], 4)
    pool = tools.CaseExamPool([train], held_out=held)
    return pool, held, train, elites


def _promote_kwargs():
    pset = gp.PrimitiveSetTyped("policy_main", [float, float], float)
    pset.add_primitive(operator.add, [float, float], float)
    tree = gp.PrimitiveTree.from_string("add(ARG0, ARG1)", pset)
    return {"prim_set": pset, "expr": tree}


def test_public_surface_has_no_push_policy_types():
    assert "PushPolicyProgram" not in gp.__all__
    assert "LinearPolicyProgram" not in gp.__all__
    assert "step_policy_loop" not in gp.__all__
    tools_source = inspect.getsource(tools)
    assert "policy_loop" not in tools_source
    assert "policy_push" not in tools_source
    assert "policy_linear" not in tools_source


def test_policy_loop_action_index_round_trip():
    for action in POLICY_LOOP_ACTIONS:
        assert POLICY_LOOP_ACTIONS[policy_action_index(action)] == action


def test_rejected_action_surfaces_in_next_observation(ind_cls):
    pool, _held, _train, elites = _pool_and_elites(ind_cls)
    guard = tools.PolicyActionGuard(max_promotes_per_gen=0)
    program = LinearPolicyProgram(
        rules=(
            PolicyDecisionRule(
                field="unsolved_count",
                op="ge",
                value=0,
                action="promote_subtree",
            ),
        ),
        default_action=tools.POLICY_ACTION_SKIP_TUNE,
    )
    step = step_policy_loop(
        lambda obs: linear_policy_decide(program, obs),
        solve_bits=(1, 0, 0, 1),
        train_score=1.0,
        action_kwargs=_promote_kwargs(),
        guard=guard,
    )
    assert step.result.rejected is True
    assert step.result.applied is False
    assert step.next_observation.last_action_rejected is True

    follow_up = step_policy_loop(
        lambda obs: linear_policy_decide(program, obs),
        solve_bits=(1, 0, 0, 1),
        train_score=1.0,
        action_kwargs={
            "exams": pool.exams,
            "elites": elites,
            "mut_prob": 0.0,
        },
        last_action_rejected=step.next_observation.last_action_rejected,
    )
    assert follow_up.observation.last_action_rejected is True
    assert follow_up.action == "promote_subtree"


def test_guard_cooldown_applies_through_policy_loop(ind_cls):
    guard = tools.PolicyActionGuard(max_promotes_per_gen=1, promote_cooldown=2)
    program = LinearPolicyProgram(
        rules=(
            PolicyDecisionRule(
                field="unsolved_count",
                op="ge",
                value=0,
                action="promote_subtree",
            ),
        ),
        default_action=tools.POLICY_ACTION_SKIP_TUNE,
    )
    kwargs = _promote_kwargs()

    first = step_policy_loop(
        lambda obs: linear_policy_decide(program, obs),
        solve_bits=(0, 1),
        train_score=1.0,
        action_kwargs=kwargs,
        guard=guard,
    )
    assert first.result.applied is True

    second = step_policy_loop(
        lambda obs: linear_policy_decide(program, obs),
        solve_bits=(0, 1),
        train_score=1.0,
        action_kwargs=kwargs,
        guard=guard,
    )
    assert second.result.rejected is True
    assert second.next_observation.last_action_rejected is True


def test_policy_loop_does_not_accept_raw_matrix_in_observe():
    matrix = numpy.zeros((2, 4))
    with pytest.raises(TypeError, match="must not be a raw array"):
        tools.policy_observe(solve_bits=matrix, train_score=0.0)  # ty: ignore[invalid-argument-type]


def test_interpret_tapes_only_via_explicit_action(ind_cls):
    matrix = numpy.array([[1.0, 2.0], [3.0, 4.0]])
    tapes = [object()]
    program = LinearPolicyProgram(
        rules=(
            PolicyDecisionRule(
                field="rows_seen",
                op="gt",
                value=0,
                action="interpret_tapes",
            ),
        ),
        default_action=tools.POLICY_ACTION_SKIP_TUNE,
    )
    with mock.patch("deap_er.private.algorithms.policy_action.interpret_tapes") as interpret:
        interpret.return_value = numpy.array([1.0, 2.0])
        step = step_policy_loop(
            lambda obs: linear_policy_decide(program, obs),
            solve_bits=(1, 1),
            train_score=0.0,
            rows_seen=1,
            action_kwargs={"tapes": tapes, "matrix": matrix},
        )
    interpret.assert_called_once()
    assert step.action == "interpret_tapes"
    assert step.result.applied is True


def test_rejected_guard_does_not_call_underlying_promote():
    guard = tools.PolicyActionGuard(max_promotes_per_gen=0)
    program = LinearPolicyProgram(
        rules=(
            PolicyDecisionRule(
                field="unsolved_count",
                op="ge",
                value=0,
                action="promote_subtree",
            ),
        ),
        default_action=tools.POLICY_ACTION_SKIP_TUNE,
    )
    with mock.patch("deap_er.private.algorithms.policy_action.promote_subtree") as promote:
        step = step_policy_loop(
            lambda obs: linear_policy_decide(program, obs),
            solve_bits=(0, 1),
            train_score=0.0,
            action_kwargs=_promote_kwargs(),
            guard=guard,
        )
    promote.assert_not_called()
    assert step.result.rejected is True


def test_step_policy_loop_docstring_notes_caller_owned_counters():
    doc = step_policy_loop.__doc__ or ""
    lowered = doc.lower()
    assert "caller" in lowered
    assert "nevals" in lowered
    assert "rows_seen" in lowered


def test_step_policy_loop_unknown_action_rejected_without_crash():
    step = step_policy_loop(
        lambda _obs: "load_column",
        solve_bits=(1,),
        train_score=0.0,
        action_kwargs={},
    )
    assert step.result.rejected is True
    assert step.result.applied is False
    assert step.next_observation.last_action_rejected is True

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
import pytest
from deap_er import Fitness, creator, tools
from deap_er.private.programming.policy_linear import (
    LinearPolicyProgram,
    PolicyDecisionRule,
    linear_policy_decide,
)
from deap_er.private.programming.policy_loop import step_policy_loop

POLICY_FIT = "P19_LINEAR_FIT"
POLICY_IND = "P19_LINEAR_IND"


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


def test_linear_policy_observe_action_round_trip(ind_cls):
    pool, _held, _train, elites = _pool_and_elites(ind_cls)
    program = LinearPolicyProgram(
        rules=(
            PolicyDecisionRule(
                field="last_action_rejected",
                op="is_true",
                action=tools.POLICY_ACTION_SKIP_TUNE,
            ),
            PolicyDecisionRule(
                field="unsolved_count",
                op="gt",
                value=1,
                action="next_lexicase_cases",
            ),
        ),
        default_action=tools.POLICY_ACTION_SKIP_PROMOTE,
    )
    step = step_policy_loop(
        lambda obs: linear_policy_decide(program, obs),
        solve_bits=(1, 0, 0, 1),
        train_score=2.0,
        held_out_score=1.0,
        action_kwargs={
            "exams": pool.exams,
            "elites": elites,
            "mut_prob": 0.0,
        },
    )
    assert step.action == "next_lexicase_cases"
    assert step.result.applied is True
    assert step.result.rejected is False
    assert step.next_observation.last_action_rejected is False
    assert isinstance(step.observation.solve_bits, tuple)
    assert step.observation.solve_bits == (1, 0, 0, 1)


def test_held_out_only_policy_fitness_after_policy_steps(ind_cls):
    pool, held, _train, elites = _pool_and_elites(ind_cls)
    program = LinearPolicyProgram(
        rules=(
            PolicyDecisionRule(
                field="solve_bit",
                solve_bit_index=2,
                op="eq",
                value=0,
                action="next_lexicase_cases",
            ),
        ),
        default_action=tools.POLICY_ACTION_SKIP_TUNE,
    )
    step_policy_loop(
        lambda obs: linear_policy_decide(program, obs),
        solve_bits=(1, 0, 0, 1),
        train_score=float(sum(tools.score_case_exams(pool.exams, elites))),
        held_out_score=float(tools.score_case_exams([held], elites)[0]),
        action_kwargs={
            "exams": pool,
            "elites": elites,
            "mut_prob": 0.0,
            "held_out": held,
        },
    )
    fitness = tools.policy_held_out_fitness(elites, pool)
    assert fitness == float(tools.score_case_exams([held], elites)[0])


def test_linear_held_out_score_none_does_not_match_zero_rule():
    program = LinearPolicyProgram(
        rules=(
            PolicyDecisionRule(
                field="held_out_score",
                op="ge",
                value=0,
                action="next_lexicase_cases",
            ),
        ),
        default_action=tools.POLICY_ACTION_SKIP_TUNE,
    )
    obs = tools.policy_observe(solve_bits=(1,), train_score=0.0, held_out_score=None)
    assert linear_policy_decide(program, obs) == tools.POLICY_ACTION_SKIP_TUNE


def test_linear_held_out_present_field():
    program = LinearPolicyProgram(
        rules=(
            PolicyDecisionRule(
                field="held_out_present",
                op="is_true",
                action="next_lexicase_cases",
            ),
        ),
        default_action=tools.POLICY_ACTION_SKIP_TUNE,
    )
    missing = tools.policy_observe(solve_bits=(1,), train_score=0.0, held_out_score=None)
    present = tools.policy_observe(solve_bits=(1,), train_score=0.0, held_out_score=2.0)
    assert linear_policy_decide(program, missing) == tools.POLICY_ACTION_SKIP_TUNE
    assert linear_policy_decide(program, present) == "next_lexicase_cases"


def test_policy_decision_rule_rejects_empty_action():
    with pytest.raises(ValueError, match="action must not be empty"):
        PolicyDecisionRule(field="unsolved_count", op="gt", value=0, action="")


def test_linear_policy_program_rejects_empty_default_action():
    with pytest.raises(ValueError, match="default_action must not be empty"):
        LinearPolicyProgram(rules=(), default_action="")


def test_multi_generation_policy_loop_tracks_caller_counters(ind_cls):
    pool, _held, _train, elites = _pool_and_elites(ind_cls)
    program = LinearPolicyProgram(
        rules=(
            PolicyDecisionRule(
                field="nevals",
                op="lt",
                value=3,
                action="next_lexicase_cases",
            ),
        ),
        default_action=tools.POLICY_ACTION_SKIP_TUNE,
    )
    nevals_seen: list[int] = []
    actions: list[str] = []

    for generation in range(4):
        step = step_policy_loop(
            lambda obs: linear_policy_decide(program, obs),
            solve_bits=(1, 0, 0, 1),
            train_score=1.0,
            nevals=generation,
            action_kwargs={
                "exams": pool.exams,
                "elites": elites,
                "mut_prob": 0.0,
            },
            last_action_rejected=generation > 0,
        )
        nevals_seen.append(step.observation.nevals)
        actions.append(step.action)

    assert nevals_seen == [0, 1, 2, 3]
    assert actions[:3] == ["next_lexicase_cases"] * 3
    assert actions[3] == tools.POLICY_ACTION_SKIP_TUNE

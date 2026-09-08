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
from deap_er.private.programming.policy_push import (
    EMIT,
    EQ,
    GT,
    LOAD_COVERAGE,
    LOAD_HELD_OUT_SET,
    LOAD_QD,
    LOAD_REJECTED,
    LOAD_SOLVE_BIT,
    LOAD_TRAIN_SCORE,
    LOAD_UNSOLVED,
    LT,
    NOT,
    PUSH_BOOL,
    PUSH_INT,
    PUSH_POLICY_OPS,
    PushPolicyProgram,
    push_policy_decide,
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
    import inspect

    import deap_er.tools as tools_module

    tools_source = inspect.getsource(tools_module)
    assert "policy_loop" not in tools_source
    assert "policy_push" not in tools_source
    assert "policy_linear" not in tools_source


def test_policy_loop_action_index_round_trip():
    for action in POLICY_LOOP_ACTIONS:
        assert POLICY_LOOP_ACTIONS[policy_action_index(action)] == action


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


def test_push_policy_emits_from_solve_bits(ind_cls):
    skip_index = policy_action_index(tools.POLICY_ACTION_SKIP_TUNE)
    program = PushPolicyProgram(
        code=(
            LOAD_UNSOLVED,
            PUSH_INT,
            2,
            LT,
            PUSH_BOOL,
            1,
            EMIT,
            skip_index,
        ),
        default_action=policy_action_index(tools.POLICY_ACTION_SKIP_PROMOTE),
    )
    obs = tools.policy_observe(solve_bits=(1, 0, 0, 1), train_score=0.0)
    assert push_policy_decide(program, obs) == tools.POLICY_ACTION_SKIP_TUNE


def test_push_policy_uses_solve_bit_index():
    next_index = policy_action_index("next_lexicase_cases")
    program = PushPolicyProgram(
        code=(
            LOAD_SOLVE_BIT,
            1,
            PUSH_INT,
            0,
            EQ,
            PUSH_BOOL,
            1,
            EMIT,
            next_index,
        ),
    )
    obs = tools.policy_observe(solve_bits=(1, 0, 1, 1), train_score=0.0)
    assert push_policy_decide(program, obs) == "next_lexicase_cases"


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


def test_push_policy_rejects_unknown_opcode():
    program = PushPolicyProgram(code=(999,))
    obs = tools.policy_observe(solve_bits=(1,), train_score=0.0)
    with pytest.raises(ValueError, match="unknown Push policy opcode"):
        push_policy_decide(program, obs)


def test_push_policy_rejects_out_of_range_emit():
    program = PushPolicyProgram(code=(EMIT, len(POLICY_LOOP_ACTIONS) + 5))
    obs = tools.policy_observe(solve_bits=(1,), train_score=0.0)
    with pytest.raises(ValueError, match="out-of-range action index"):
        push_policy_decide(program, obs)


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


def test_push_policy_conditional_emit_respects_false_branch():
    skip_index = policy_action_index(tools.POLICY_ACTION_SKIP_TUNE)
    next_index = policy_action_index("next_lexicase_cases")
    program = PushPolicyProgram(
        code=(
            LOAD_REJECTED,
            EMIT,
            next_index,
            LOAD_REJECTED,
            NOT,
            EMIT,
            skip_index,
        ),
        default_action=skip_index,
    )
    obs = tools.policy_observe(
        solve_bits=(1,),
        train_score=0.0,
        last_action_rejected=False,
    )
    assert push_policy_decide(program, obs) == tools.POLICY_ACTION_SKIP_TUNE

    rejected = tools.policy_observe(
        solve_bits=(1,),
        train_score=0.0,
        last_action_rejected=True,
    )
    assert push_policy_decide(program, rejected) == "next_lexicase_cases"


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


def test_push_policy_loads_summary_scalars():
    next_index = policy_action_index("next_lexicase_cases")
    skip_index = policy_action_index(tools.POLICY_ACTION_SKIP_TUNE)
    program = PushPolicyProgram(
        code=(
            LOAD_TRAIN_SCORE,
            PUSH_INT,
            1,
            GT,
            PUSH_BOOL,
            1,
            EMIT,
            next_index,
        ),
        default_action=skip_index,
    )
    obs = tools.policy_observe(
        solve_bits=(1,),
        train_score=2.5,
        archive=tools.ArchiveStats(
            num_elites=2,
            num_cells=4,
            coverage=0.5,
            qd_score=3.0,
        ),
    )
    assert push_policy_decide(program, obs) == "next_lexicase_cases"


def test_push_policy_loads_archive_coverage():
    next_index = policy_action_index("next_lexicase_cases")
    skip_index = policy_action_index(tools.POLICY_ACTION_SKIP_TUNE)
    program = PushPolicyProgram(
        code=(
            LOAD_COVERAGE,
            PUSH_INT,
            0,
            GT,
            PUSH_BOOL,
            1,
            EMIT,
            next_index,
        ),
        default_action=skip_index,
    )
    obs = tools.policy_observe(
        solve_bits=(1,),
        train_score=0.0,
        archive=tools.ArchiveStats(
            num_elites=2,
            num_cells=4,
            coverage=0.5,
            qd_score=3.0,
        ),
    )
    assert push_policy_decide(program, obs) == "next_lexicase_cases"


def test_push_policy_loads_held_out_when_present():
    next_index = policy_action_index("next_lexicase_cases")
    skip_index = policy_action_index(tools.POLICY_ACTION_SKIP_TUNE)
    program = PushPolicyProgram(
        code=(
            LOAD_HELD_OUT_SET,
            EMIT,
            next_index,
            LOAD_HELD_OUT_SET,
            NOT,
            EMIT,
            skip_index,
        ),
        default_action=skip_index,
    )
    missing = tools.policy_observe(solve_bits=(1,), train_score=0.0, held_out_score=None)
    present = tools.policy_observe(solve_bits=(1,), train_score=0.0, held_out_score=1.0)
    assert push_policy_decide(program, missing) == tools.POLICY_ACTION_SKIP_TUNE
    assert push_policy_decide(program, present) == "next_lexicase_cases"


def test_push_policy_loads_qd_score():
    next_index = policy_action_index("next_lexicase_cases")
    program = PushPolicyProgram(
        code=(
            LOAD_QD,
            PUSH_INT,
            2,
            GT,
            PUSH_BOOL,
            1,
            EMIT,
            next_index,
        ),
    )
    obs = tools.policy_observe(
        solve_bits=(1,),
        train_score=0.0,
        archive=tools.ArchiveStats(
            num_elites=2,
            num_cells=4,
            coverage=0.5,
            qd_score=3.0,
        ),
    )
    assert push_policy_decide(program, obs) == "next_lexicase_cases"


def test_push_policy_truncated_program_raises_value_error():
    program = PushPolicyProgram(code=(PUSH_INT,))
    obs = tools.policy_observe(solve_bits=(1,), train_score=0.0)
    with pytest.raises(ValueError, match="truncated Push policy program"):
        push_policy_decide(program, obs)


def test_push_policy_last_emit_wins():
    skip_index = policy_action_index(tools.POLICY_ACTION_SKIP_TUNE)
    next_index = policy_action_index("next_lexicase_cases")
    program = PushPolicyProgram(
        code=(
            EMIT,
            next_index,
            EMIT,
            skip_index,
        ),
        default_action=next_index,
    )
    obs = tools.policy_observe(solve_bits=(1,), train_score=0.0)
    assert push_policy_decide(program, obs) == tools.POLICY_ACTION_SKIP_TUNE


def test_push_policy_forbidden_opcodes_absent():
    import deap_er.private.programming.policy_push as policy_push

    forbidden_names = (
        "LOAD_COLUMN",
        "LOAD_MATRIX",
        "LOAD_WINDOW",
        "ROLLING_MEAN",
        "INTERPRET_TAPE",
    )
    for name in forbidden_names:
        assert not hasattr(policy_push, name)
    assert max(PUSH_POLICY_OPS) < 100


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

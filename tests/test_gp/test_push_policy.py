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
import deap_er.private.programming.policy_push as policy_push
import pytest
from deap_er import tools
from deap_er.private.programming.policy_loop import POLICY_LOOP_ACTIONS, policy_action_index


def test_push_policy_emits_from_solve_bits():
    skip_index = policy_action_index(tools.POLICY_ACTION_SKIP_TUNE)
    program = policy_push.PushPolicyProgram(
        code=(
            policy_push.LOAD_UNSOLVED,
            policy_push.PUSH_INT,
            2,
            policy_push.LT,
            policy_push.PUSH_BOOL,
            1,
            policy_push.EMIT,
            skip_index,
        ),
        default_action=policy_action_index(tools.POLICY_ACTION_SKIP_PROMOTE),
    )
    obs = tools.policy_observe(solve_bits=(1, 0, 0, 1), train_score=0.0)
    assert policy_push.push_policy_decide(program, obs) == tools.POLICY_ACTION_SKIP_TUNE


def test_push_policy_uses_solve_bit_index():
    next_index = policy_action_index("next_lexicase_cases")
    program = policy_push.PushPolicyProgram(
        code=(
            policy_push.LOAD_SOLVE_BIT,
            1,
            policy_push.PUSH_INT,
            0,
            policy_push.EQ,
            policy_push.PUSH_BOOL,
            1,
            policy_push.EMIT,
            next_index,
        ),
    )
    obs = tools.policy_observe(solve_bits=(1, 0, 1, 1), train_score=0.0)
    assert policy_push.push_policy_decide(program, obs) == "next_lexicase_cases"


def test_push_policy_rejects_unknown_opcode():
    program = policy_push.PushPolicyProgram(code=(999,))
    obs = tools.policy_observe(solve_bits=(1,), train_score=0.0)
    with pytest.raises(ValueError, match="unknown Push policy opcode"):
        policy_push.push_policy_decide(program, obs)


def test_push_policy_rejects_out_of_range_emit():
    program = policy_push.PushPolicyProgram(code=(policy_push.EMIT, len(POLICY_LOOP_ACTIONS) + 5))
    obs = tools.policy_observe(solve_bits=(1,), train_score=0.0)
    with pytest.raises(ValueError, match="out-of-range action index"):
        policy_push.push_policy_decide(program, obs)


def test_push_policy_conditional_emit_respects_false_branch():
    skip_index = policy_action_index(tools.POLICY_ACTION_SKIP_TUNE)
    next_index = policy_action_index("next_lexicase_cases")
    program = policy_push.PushPolicyProgram(
        code=(
            policy_push.LOAD_REJECTED,
            policy_push.EMIT,
            next_index,
            policy_push.LOAD_REJECTED,
            policy_push.NOT,
            policy_push.EMIT,
            skip_index,
        ),
        default_action=skip_index,
    )
    obs = tools.policy_observe(
        solve_bits=(1,),
        train_score=0.0,
        last_action_rejected=False,
    )
    assert policy_push.push_policy_decide(program, obs) == tools.POLICY_ACTION_SKIP_TUNE

    rejected = tools.policy_observe(
        solve_bits=(1,),
        train_score=0.0,
        last_action_rejected=True,
    )
    assert policy_push.push_policy_decide(program, rejected) == "next_lexicase_cases"


def test_push_policy_loads_summary_scalars():
    next_index = policy_action_index("next_lexicase_cases")
    skip_index = policy_action_index(tools.POLICY_ACTION_SKIP_TUNE)
    program = policy_push.PushPolicyProgram(
        code=(
            policy_push.LOAD_TRAIN_SCORE,
            policy_push.PUSH_INT,
            1,
            policy_push.GT,
            policy_push.PUSH_BOOL,
            1,
            policy_push.EMIT,
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
    assert policy_push.push_policy_decide(program, obs) == "next_lexicase_cases"


def test_push_policy_loads_archive_coverage():
    next_index = policy_action_index("next_lexicase_cases")
    skip_index = policy_action_index(tools.POLICY_ACTION_SKIP_TUNE)
    program = policy_push.PushPolicyProgram(
        code=(
            policy_push.LOAD_COVERAGE,
            policy_push.PUSH_INT,
            0,
            policy_push.GT,
            policy_push.PUSH_BOOL,
            1,
            policy_push.EMIT,
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
    assert policy_push.push_policy_decide(program, obs) == "next_lexicase_cases"


def test_push_policy_loads_held_out_when_present():
    next_index = policy_action_index("next_lexicase_cases")
    skip_index = policy_action_index(tools.POLICY_ACTION_SKIP_TUNE)
    program = policy_push.PushPolicyProgram(
        code=(
            policy_push.LOAD_HELD_OUT_SET,
            policy_push.EMIT,
            next_index,
            policy_push.LOAD_HELD_OUT_SET,
            policy_push.NOT,
            policy_push.EMIT,
            skip_index,
        ),
        default_action=skip_index,
    )
    missing = tools.policy_observe(solve_bits=(1,), train_score=0.0, held_out_score=None)
    present = tools.policy_observe(solve_bits=(1,), train_score=0.0, held_out_score=1.0)
    assert policy_push.push_policy_decide(program, missing) == tools.POLICY_ACTION_SKIP_TUNE
    assert policy_push.push_policy_decide(program, present) == "next_lexicase_cases"


def test_push_policy_loads_qd_score():
    next_index = policy_action_index("next_lexicase_cases")
    program = policy_push.PushPolicyProgram(
        code=(
            policy_push.LOAD_QD,
            policy_push.PUSH_INT,
            2,
            policy_push.GT,
            policy_push.PUSH_BOOL,
            1,
            policy_push.EMIT,
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
    assert policy_push.push_policy_decide(program, obs) == "next_lexicase_cases"


def test_push_policy_truncated_program_raises_value_error():
    program = policy_push.PushPolicyProgram(code=(policy_push.PUSH_INT,))
    obs = tools.policy_observe(solve_bits=(1,), train_score=0.0)
    with pytest.raises(ValueError, match="truncated Push policy program"):
        policy_push.push_policy_decide(program, obs)


def test_push_policy_last_emit_wins():
    skip_index = policy_action_index(tools.POLICY_ACTION_SKIP_TUNE)
    next_index = policy_action_index("next_lexicase_cases")
    program = policy_push.PushPolicyProgram(
        code=(
            policy_push.EMIT,
            next_index,
            policy_push.EMIT,
            skip_index,
        ),
        default_action=next_index,
    )
    obs = tools.policy_observe(solve_bits=(1,), train_score=0.0)
    assert policy_push.push_policy_decide(program, obs) == tools.POLICY_ACTION_SKIP_TUNE


def test_push_policy_forbidden_opcodes_absent():
    forbidden_names = (
        "LOAD_COLUMN",
        "LOAD_MATRIX",
        "LOAD_WINDOW",
        "ROLLING_MEAN",
        "INTERPRET_TAPE",
    )
    for name in forbidden_names:
        assert not hasattr(policy_push, name)
    assert max(policy_push.PUSH_POLICY_OPS) < 100

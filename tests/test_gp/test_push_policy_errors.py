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
from deap_er.private.programming.policy_loop import POLICY_LOOP_ACTIONS


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


def test_push_policy_truncated_program_raises_value_error():
    program = policy_push.PushPolicyProgram(code=(policy_push.PUSH_INT,))
    obs = tools.policy_observe(solve_bits=(1,), train_score=0.0)
    with pytest.raises(ValueError, match="truncated Push policy program"):
        policy_push.push_policy_decide(program, obs)


def test_push_policy_stack_underflow_raises_value_error():
    program = policy_push.PushPolicyProgram(code=(policy_push.PUSH_INT, 1, policy_push.SUB))
    obs = tools.policy_observe(solve_bits=(1,), train_score=0.0)
    with pytest.raises(ValueError, match="stack underflow"):
        policy_push.push_policy_decide(program, obs)


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

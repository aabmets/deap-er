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
from __future__ import annotations

from dataclasses import dataclass

from deap_er.private.programming.policy_loop import (
    POLICY_LOOP_ACTIONS,
    policy_action_from_index,
)
from deap_er.private.programming.policy_push_ops import (
    ADD,
    AND,
    EMIT,
    EQ,
    GT,
    LOAD_COVERAGE,
    LOAD_HELD_OUT,
    LOAD_HELD_OUT_SET,
    LOAD_INVALID,
    LOAD_NE_EVALS,
    LOAD_PROMOTED,
    LOAD_QD,
    LOAD_REJECTED,
    LOAD_ROWS,
    LOAD_SOLVE_BIT,
    LOAD_TRAIN_SCORE,
    LOAD_UNSOLVED,
    LT,
    NOT,
    OR,
    PUSH_BOOL,
    PUSH_INT,
    PUSH_POLICY_OPS,
    SUB,
    PushPolicyOp,
    StackValue,
)
from deap_er.private.records.policy_observation import PolicyObservation

__all__: list[str] = [
    "ADD",
    "AND",
    "EMIT",
    "EQ",
    "GT",
    "LOAD_COVERAGE",
    "LOAD_HELD_OUT",
    "LOAD_HELD_OUT_SET",
    "LOAD_INVALID",
    "LOAD_NE_EVALS",
    "LOAD_PROMOTED",
    "LOAD_QD",
    "LOAD_REJECTED",
    "LOAD_ROWS",
    "LOAD_SOLVE_BIT",
    "LOAD_TRAIN_SCORE",
    "LOAD_UNSOLVED",
    "LT",
    "NOT",
    "OR",
    "PUSH_BOOL",
    "PUSH_INT",
    "PUSH_POLICY_OPS",
    "PushPolicyOp",
    "PushPolicyProgram",
    "StackValue",
    "SUB",
    "push_policy_decide",
]


@dataclass(frozen=True, slots=True)
class PushPolicyProgram:
    """Private Push policy individual as a flat instruction tape.

    Attributes:
        code: Alternating opcodes and literal operands. Literals follow
            :data:`PUSH_INT`, :data:`PUSH_BOOL`, :data:`LOAD_SOLVE_BIT`,
            and :data:`EMIT`.
        default_action: Action index used when the program never emits.
            When several :data:`EMIT` instructions run, the last
            successful emit wins.
    """

    code: tuple[int, ...]
    default_action: int = 0


def push_policy_decide(
    program: PushPolicyProgram,
    observation: PolicyObservation,
) -> str:
    """Interpret a tiny Push program on one observation summary.

    Args:
        program: Private Push instruction tape.
        observation: Summary observation from
            :func:`~deap_er.tools.policy_observe`.

    Returns:
        A token from :data:`~deap_er.private.programming.policy_loop.POLICY_LOOP_ACTIONS`.

    Raises:
        ValueError: If the program is truncated, references an unknown
            opcode, or emits an invalid action index.
    """
    stack: list[StackValue] = []
    emitted: int | None = None
    code = program.code
    index = 0
    while index < len(code):
        opcode = code[index]
        index += 1
        if opcode not in PUSH_POLICY_OPS:
            raise ValueError(f"unknown Push policy opcode: {opcode}")
        if opcode == PUSH_INT:
            value, index = _operand(code, index)
            stack.append(value)
            continue
        if opcode == PUSH_BOOL:
            value, index = _operand(code, index)
            stack.append(bool(value))
            continue
        if opcode == LOAD_UNSOLVED:
            stack.append(observation.unsolved_count)
            continue
        if opcode == LOAD_NE_EVALS:
            stack.append(observation.nevals)
            continue
        if opcode == LOAD_ROWS:
            stack.append(observation.rows_seen)
            continue
        if opcode == LOAD_PROMOTED:
            stack.append(observation.promoted_library_size)
            continue
        if opcode == LOAD_INVALID:
            stack.append(observation.fitness_invalid)
            continue
        if opcode == LOAD_REJECTED:
            stack.append(observation.last_action_rejected)
            continue
        if opcode == LOAD_TRAIN_SCORE:
            stack.append(observation.train_score)
            continue
        if opcode == LOAD_HELD_OUT:
            if observation.held_out_score is None:
                raise ValueError("held_out_score is not available")
            stack.append(observation.held_out_score)
            continue
        if opcode == LOAD_HELD_OUT_SET:
            stack.append(observation.held_out_score is not None)
            continue
        if opcode == LOAD_COVERAGE:
            stack.append(observation.archive_coverage)
            continue
        if opcode == LOAD_QD:
            stack.append(observation.qd_score)
            continue
        if opcode == LOAD_SOLVE_BIT:
            bit_index, index = _operand(code, index)
            bits = observation.solve_bits
            value = 0
            if 0 <= bit_index < len(bits):
                value = bits[bit_index]
            stack.append(value)
            continue
        if opcode == ADD:
            right = _pop_int(stack)
            left = _pop_int(stack)
            stack.append(left + right)
            continue
        if opcode == SUB:
            right = _pop_int(stack)
            left = _pop_int(stack)
            stack.append(left - right)
            continue
        if opcode == LT:
            right = _pop_numeric(stack)
            left = _pop_numeric(stack)
            stack.append(left < right)
            continue
        if opcode == GT:
            right = _pop_numeric(stack)
            left = _pop_numeric(stack)
            stack.append(left > right)
            continue
        if opcode == EQ:
            right = stack.pop()
            left = stack.pop()
            stack.append(left == right)
            continue
        if opcode == AND:
            right = _pop_bool(stack)
            left = _pop_bool(stack)
            stack.append(left and right)
            continue
        if opcode == OR:
            right = _pop_bool(stack)
            left = _pop_bool(stack)
            stack.append(left or right)
            continue
        if opcode == NOT:
            stack.append(not _pop_bool(stack))
            continue
        if opcode == EMIT:
            action_index, index = _operand(code, index)
            if stack and isinstance(stack[-1], bool):
                condition = stack.pop()
                if condition:
                    emitted = action_index
            else:
                emitted = action_index
            continue
    if emitted is None:
        emitted = program.default_action
    if emitted < 0 or emitted >= len(POLICY_LOOP_ACTIONS):
        raise ValueError("Push policy emitted an out-of-range action index")
    return policy_action_from_index(emitted)


def _operand(code: tuple[int, ...], index: int) -> tuple[int, int]:
    if index >= len(code):
        raise ValueError("truncated Push policy program")
    return code[index], index + 1


def _pop_int(stack: list[StackValue]) -> int:
    value = stack.pop()
    if isinstance(value, bool):
        return int(value)
    return int(value)


def _pop_numeric(stack: list[StackValue]) -> int | float:
    value = stack.pop()
    if isinstance(value, bool):
        return int(value)
    return value


def _pop_bool(stack: list[StackValue]) -> bool:
    value = stack.pop()
    if isinstance(value, bool):
        return value
    return bool(value)

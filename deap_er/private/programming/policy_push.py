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
from deap_er.private.records.policy_observation import PolicyObservation

__all__: list[str] = [
    "PUSH_POLICY_OPS",
    "PushPolicyOp",
    "PushPolicyProgram",
    "push_policy_decide",
]

PUSH_INT = 1
LOAD_UNSOLVED = 2
LOAD_NE_EVALS = 3
LOAD_ROWS = 4
LOAD_PROMOTED = 5
LOAD_INVALID = 6
LOAD_REJECTED = 7
LOAD_SOLVE_BIT = 8
PUSH_BOOL = 9
ADD = 10
SUB = 11
LT = 12
GT = 13
EQ = 14
AND = 15
OR = 16
NOT = 17
EMIT = 18

PUSH_POLICY_OPS: frozenset[int] = frozenset(
    {
        PUSH_INT,
        LOAD_UNSOLVED,
        LOAD_NE_EVALS,
        LOAD_ROWS,
        LOAD_PROMOTED,
        LOAD_INVALID,
        LOAD_REJECTED,
        LOAD_SOLVE_BIT,
        PUSH_BOOL,
        ADD,
        SUB,
        LT,
        GT,
        EQ,
        AND,
        OR,
        NOT,
        EMIT,
    }
)


@dataclass(frozen=True, slots=True)
class PushPolicyOp:
    """Named constants for the private Push policy instruction set.

    The interpreter only manipulates ``int`` and ``bool`` stack values
    plus short solve-bit vectors read through :data:`LOAD_SOLVE_BIT`.
    There is no column load, no ``Window`` type, and no tape opcode.
    """

    PUSH_INT: int = PUSH_INT
    LOAD_UNSOLVED: int = LOAD_UNSOLVED
    LOAD_NE_EVALS: int = LOAD_NE_EVALS
    LOAD_ROWS: int = LOAD_ROWS
    LOAD_PROMOTED: int = LOAD_PROMOTED
    LOAD_INVALID: int = LOAD_INVALID
    LOAD_REJECTED: int = LOAD_REJECTED
    LOAD_SOLVE_BIT: int = LOAD_SOLVE_BIT
    PUSH_BOOL: int = PUSH_BOOL
    ADD: int = ADD
    SUB: int = SUB
    LT: int = LT
    GT: int = GT
    EQ: int = EQ
    AND: int = AND
    OR: int = OR
    NOT: int = NOT
    EMIT: int = EMIT


@dataclass(frozen=True, slots=True)
class PushPolicyProgram:
    """Private Push policy individual as a flat instruction tape.

    Attributes:
        code: Alternating opcodes and literal operands. Literals follow
            :data:`PUSH_INT`, :data:`PUSH_BOOL`, :data:`LOAD_SOLVE_BIT`,
            and :data:`EMIT`.
        default_action: Action index used when the program never emits.
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
        ValueError: If the program references an unknown opcode or an
            invalid action index.
    """
    stack: list[int | bool] = []
    emitted: int | None = None
    code = program.code
    index = 0
    while index < len(code):
        opcode = code[index]
        index += 1
        if opcode not in PUSH_POLICY_OPS:
            raise ValueError(f"unknown Push policy opcode: {opcode}")
        if opcode == PUSH_INT:
            stack.append(code[index])
            index += 1
            continue
        if opcode == PUSH_BOOL:
            stack.append(bool(code[index]))
            index += 1
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
        if opcode == LOAD_SOLVE_BIT:
            bit_index = code[index]
            index += 1
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
            right = _pop_int(stack)
            left = _pop_int(stack)
            stack.append(left < right)
            continue
        if opcode == GT:
            right = _pop_int(stack)
            left = _pop_int(stack)
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
            action_index = code[index]
            index += 1
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


def _pop_int(stack: list[int | bool]) -> int:
    value = stack.pop()
    if isinstance(value, bool):
        return int(value)
    return int(value)


def _pop_bool(stack: list[int | bool]) -> bool:
    value = stack.pop()
    if isinstance(value, bool):
        return value
    return bool(value)

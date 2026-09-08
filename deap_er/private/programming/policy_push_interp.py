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

from collections.abc import Callable
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
    StackValue,
)
from deap_er.private.records.policy_observation import PolicyObservation

__all__: list[str] = ["interpret_push_policy"]


@dataclass
class _PushInterpState:
    code: tuple[int, ...]
    index: int
    stack: list[StackValue]
    emitted: int | None
    observation: PolicyObservation


def interpret_push_policy(
    code: tuple[int, ...],
    observation: PolicyObservation,
    *,
    default_action: int,
) -> str:
    """Run a Push policy tape and return the chosen action token."""
    state = _PushInterpState(
        code=code,
        index=0,
        stack=[],
        emitted=None,
        observation=observation,
    )
    while state.index < len(state.code):
        opcode = state.code[state.index]
        state.index += 1
        if opcode not in PUSH_POLICY_OPS:
            raise ValueError(f"unknown Push policy opcode: {opcode}")
        _OPCODE_HANDLERS[opcode](state)
    emitted = state.emitted if state.emitted is not None else default_action
    if emitted < 0 or emitted >= len(POLICY_LOOP_ACTIONS):
        raise ValueError("Push policy emitted an out-of-range action index")
    return policy_action_from_index(emitted)


def _operand(state: _PushInterpState) -> int:
    if state.index >= len(state.code):
        raise ValueError("truncated Push policy program")
    value = state.code[state.index]
    state.index += 1
    return value


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


def _push_int(state: _PushInterpState) -> None:
    state.stack.append(_operand(state))


def _push_bool(state: _PushInterpState) -> None:
    state.stack.append(bool(_operand(state)))


def _load_unsolved(state: _PushInterpState) -> None:
    state.stack.append(state.observation.unsolved_count)


def _load_nevals(state: _PushInterpState) -> None:
    state.stack.append(state.observation.nevals)


def _load_rows(state: _PushInterpState) -> None:
    state.stack.append(state.observation.rows_seen)


def _load_promoted(state: _PushInterpState) -> None:
    state.stack.append(state.observation.promoted_library_size)


def _load_invalid(state: _PushInterpState) -> None:
    state.stack.append(state.observation.fitness_invalid)


def _load_rejected(state: _PushInterpState) -> None:
    state.stack.append(state.observation.last_action_rejected)


def _load_train_score(state: _PushInterpState) -> None:
    state.stack.append(state.observation.train_score)


def _load_held_out(state: _PushInterpState) -> None:
    if state.observation.held_out_score is None:
        raise ValueError("held_out_score is not available")
    state.stack.append(state.observation.held_out_score)


def _load_held_out_set(state: _PushInterpState) -> None:
    state.stack.append(state.observation.held_out_score is not None)


def _load_coverage(state: _PushInterpState) -> None:
    state.stack.append(state.observation.archive_coverage)


def _load_qd(state: _PushInterpState) -> None:
    state.stack.append(state.observation.qd_score)


def _load_solve_bit(state: _PushInterpState) -> None:
    bit_index = _operand(state)
    bits = state.observation.solve_bits
    value = bits[bit_index] if 0 <= bit_index < len(bits) else 0
    state.stack.append(value)


def _add(state: _PushInterpState) -> None:
    right = _pop_int(state.stack)
    left = _pop_int(state.stack)
    state.stack.append(left + right)


def _sub(state: _PushInterpState) -> None:
    right = _pop_int(state.stack)
    left = _pop_int(state.stack)
    state.stack.append(left - right)


def _lt(state: _PushInterpState) -> None:
    right = _pop_numeric(state.stack)
    left = _pop_numeric(state.stack)
    state.stack.append(left < right)


def _gt(state: _PushInterpState) -> None:
    right = _pop_numeric(state.stack)
    left = _pop_numeric(state.stack)
    state.stack.append(left > right)


def _eq(state: _PushInterpState) -> None:
    right = state.stack.pop()
    left = state.stack.pop()
    state.stack.append(left == right)


def _and(state: _PushInterpState) -> None:
    right = _pop_bool(state.stack)
    left = _pop_bool(state.stack)
    state.stack.append(left and right)


def _or(state: _PushInterpState) -> None:
    right = _pop_bool(state.stack)
    left = _pop_bool(state.stack)
    state.stack.append(left or right)


def _not(state: _PushInterpState) -> None:
    state.stack.append(not _pop_bool(state.stack))


def _emit(state: _PushInterpState) -> None:
    action_index = _operand(state)
    if state.stack and isinstance(state.stack[-1], bool):
        condition = state.stack.pop()
        if condition:
            state.emitted = action_index
        return
    state.emitted = action_index


_OPCODE_HANDLERS: dict[int, Callable[[_PushInterpState], None]] = {
    PUSH_INT: _push_int,
    PUSH_BOOL: _push_bool,
    LOAD_UNSOLVED: _load_unsolved,
    LOAD_NE_EVALS: _load_nevals,
    LOAD_ROWS: _load_rows,
    LOAD_PROMOTED: _load_promoted,
    LOAD_INVALID: _load_invalid,
    LOAD_REJECTED: _load_rejected,
    LOAD_TRAIN_SCORE: _load_train_score,
    LOAD_HELD_OUT: _load_held_out,
    LOAD_HELD_OUT_SET: _load_held_out_set,
    LOAD_COVERAGE: _load_coverage,
    LOAD_QD: _load_qd,
    LOAD_SOLVE_BIT: _load_solve_bit,
    ADD: _add,
    SUB: _sub,
    LT: _lt,
    GT: _gt,
    EQ: _eq,
    AND: _and,
    OR: _or,
    NOT: _not,
    EMIT: _emit,
}

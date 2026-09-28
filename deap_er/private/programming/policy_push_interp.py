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

from deap_er.private.records.policy_observation import PolicyObservation

from .policy_loop import (
    POLICY_LOOP_ACTIONS,
    policy_action_from_index,
)
from .policy_push_loads import PUSH_HANDLERS
from .policy_push_ops import (
    ADD,
    AND,
    EMIT,
    EQ,
    GT,
    LT,
    NOT,
    OR,
    PUSH_POLICY_OPS,
    SUB,
    PushInterpState,
    StackValue,
    read_operand,
)

__all__: list[str] = ["interpret_push_policy"]


def interpret_push_policy(
    code: tuple[int, ...],
    observation: PolicyObservation,
    *,
    default_action: int,
) -> str:
    """Run a Push policy tape and return the chosen action token."""
    state = PushInterpState(
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


def _pop(stack: list[StackValue]) -> StackValue:
    if not stack:
        raise ValueError("Push policy stack underflow")
    return stack.pop()


def _pop_numeric(stack: list[StackValue]) -> int | float:
    value = _pop(stack)
    if isinstance(value, bool):
        return int(value)
    return value


def _pop_bool(stack: list[StackValue]) -> bool:
    return bool(_pop(stack))


def _add(state: PushInterpState) -> None:
    right = _pop_numeric(state.stack)
    left = _pop_numeric(state.stack)
    state.stack.append(left + right)


def _sub(state: PushInterpState) -> None:
    right = _pop_numeric(state.stack)
    left = _pop_numeric(state.stack)
    state.stack.append(left - right)


def _lt(state: PushInterpState) -> None:
    right = _pop_numeric(state.stack)
    left = _pop_numeric(state.stack)
    state.stack.append(left < right)


def _gt(state: PushInterpState) -> None:
    right = _pop_numeric(state.stack)
    left = _pop_numeric(state.stack)
    state.stack.append(left > right)


def _eq(state: PushInterpState) -> None:
    right = _pop(state.stack)
    left = _pop(state.stack)
    state.stack.append(left == right)


def _and(state: PushInterpState) -> None:
    right = _pop_bool(state.stack)
    left = _pop_bool(state.stack)
    state.stack.append(left and right)


def _or(state: PushInterpState) -> None:
    right = _pop_bool(state.stack)
    left = _pop_bool(state.stack)
    state.stack.append(left or right)


def _not(state: PushInterpState) -> None:
    state.stack.append(not _pop_bool(state.stack))


def _emit(state: PushInterpState) -> None:
    action_index = read_operand(state)
    if state.stack and isinstance(state.stack[-1], bool):
        condition = state.stack.pop()
        if condition:
            state.emitted = action_index
        return
    state.emitted = action_index


_OPCODE_HANDLERS: dict[int, Callable[[PushInterpState], None]] = {
    **PUSH_HANDLERS,
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

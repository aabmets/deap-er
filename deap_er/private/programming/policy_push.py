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

from deap_er.private.programming.policy_push_interp import interpret_push_policy
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
    return interpret_push_policy(
        program.code,
        observation,
        default_action=program.default_action,
    )

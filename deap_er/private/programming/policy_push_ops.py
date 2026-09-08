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
    "StackValue",
    "SUB",
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
LOAD_TRAIN_SCORE = 19
LOAD_HELD_OUT = 20
LOAD_HELD_OUT_SET = 21
LOAD_COVERAGE = 22
LOAD_QD = 23

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
        LOAD_TRAIN_SCORE,
        LOAD_HELD_OUT,
        LOAD_HELD_OUT_SET,
        LOAD_COVERAGE,
        LOAD_QD,
    }
)

StackValue = int | bool | float


@dataclass(frozen=True, slots=True)
class PushPolicyOp:
    """Named constants for the private Push policy instruction set.

    The interpreter only manipulates ``int``, ``bool``, and ``float``
    stack values plus short solve-bit vectors read through
    :data:`LOAD_SOLVE_BIT`. There is no column load, no ``Window``
    type, and no tape opcode.
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
    LOAD_TRAIN_SCORE: int = LOAD_TRAIN_SCORE
    LOAD_HELD_OUT: int = LOAD_HELD_OUT
    LOAD_HELD_OUT_SET: int = LOAD_HELD_OUT_SET
    LOAD_COVERAGE: int = LOAD_COVERAGE
    LOAD_QD: int = LOAD_QD

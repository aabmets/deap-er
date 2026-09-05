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
from ..opcodes import USER_BASE, Opcode

__all__: list[str] = [
    "COL_LOAD",
    "CONST",
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "NEG",
    "ABS",
    "LOG",
    "SQRT",
    "SIN",
    "COS",
    "GT",
    "LT",
    "GE",
    "LE",
    "EQ",
    "AND",
    "OR",
    "NOT",
    "WHERE",
    "DELAY",
    "DIFF",
    "ROLL_SUM",
    "ROLL_MEAN",
    "ROLL_STD",
    "ROLL_MIN",
    "ROLL_MAX",
    "EMA",
    "BASE",
    "UNKNOWN_OPCODE",
]

COL_LOAD = int(Opcode.COL_LOAD)
CONST = int(Opcode.CONST)
ADD = int(Opcode.ADD)
SUB = int(Opcode.SUB)
MUL = int(Opcode.MUL)
DIV = int(Opcode.DIV)
NEG = int(Opcode.NEG)
ABS = int(Opcode.ABS)
LOG = int(Opcode.LOG)
SQRT = int(Opcode.SQRT)
SIN = int(Opcode.SIN)
COS = int(Opcode.COS)
GT = int(Opcode.GT)
LT = int(Opcode.LT)
GE = int(Opcode.GE)
LE = int(Opcode.LE)
EQ = int(Opcode.EQ)
AND = int(Opcode.AND)
OR = int(Opcode.OR)
NOT = int(Opcode.NOT)
WHERE = int(Opcode.WHERE)
DELAY = int(Opcode.DELAY)
DIFF = int(Opcode.DIFF)
ROLL_SUM = int(Opcode.ROLL_SUM)
ROLL_MEAN = int(Opcode.ROLL_MEAN)
ROLL_STD = int(Opcode.ROLL_STD)
ROLL_MIN = int(Opcode.ROLL_MIN)
ROLL_MAX = int(Opcode.ROLL_MAX)
EMA = int(Opcode.EMA)
BASE = USER_BASE
UNKNOWN_OPCODE = "The tape holds an opcode the interpreter does not know."

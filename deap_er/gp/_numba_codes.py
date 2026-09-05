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
from .opcodes import USER_BASE, Opcode

__all__: list[str] = []

_COL_LOAD = int(Opcode.COL_LOAD)
_CONST = int(Opcode.CONST)
_ADD = int(Opcode.ADD)
_SUB = int(Opcode.SUB)
_MUL = int(Opcode.MUL)
_DIV = int(Opcode.DIV)
_NEG = int(Opcode.NEG)
_ABS = int(Opcode.ABS)
_LOG = int(Opcode.LOG)
_SQRT = int(Opcode.SQRT)
_SIN = int(Opcode.SIN)
_COS = int(Opcode.COS)
_GT = int(Opcode.GT)
_LT = int(Opcode.LT)
_GE = int(Opcode.GE)
_LE = int(Opcode.LE)
_EQ = int(Opcode.EQ)
_AND = int(Opcode.AND)
_OR = int(Opcode.OR)
_NOT = int(Opcode.NOT)
_WHERE = int(Opcode.WHERE)
_DELAY = int(Opcode.DELAY)
_DIFF = int(Opcode.DIFF)
_ROLL_SUM = int(Opcode.ROLL_SUM)
_ROLL_MEAN = int(Opcode.ROLL_MEAN)
_ROLL_STD = int(Opcode.ROLL_STD)
_ROLL_MIN = int(Opcode.ROLL_MIN)
_ROLL_MAX = int(Opcode.ROLL_MAX)
_EMA = int(Opcode.EMA)
_BASE = USER_BASE

_UNKNOWN_OPCODE = "The tape holds an opcode the interpreter does not know."

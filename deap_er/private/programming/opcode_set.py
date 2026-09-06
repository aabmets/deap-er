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
from enum import IntEnum

__all__: list[str] = ["USER_BASE", "Opcode", "BUILTIN_OPCODES", "OPCODES_ARITY"]

USER_BASE = 1000
"""First opcode value reserved for consumer-supplied kernels."""


class Opcode(IntEnum):
    """Instruction set of the tree stack machine.

    Every value below ``USER_BASE`` is owned by this library. Consumers
    bind their own kernels to values at or above ``USER_BASE``.
    """

    COL_LOAD = 0
    CONST = 1
    ADD = 2
    SUB = 3
    MUL = 4
    DIV = 5
    NEG = 6
    ABS = 7
    LOG = 8
    SQRT = 9
    SIN = 10
    COS = 11
    GT = 12
    LT = 13
    GE = 14
    LE = 15
    EQ = 16
    AND = 17
    OR = 18
    NOT = 19
    WHERE = 20
    DELAY = 21
    DIFF = 22
    ROLL_SUM = 23
    ROLL_MEAN = 24
    ROLL_STD = 25
    ROLL_MIN = 26
    ROLL_MAX = 27
    EMA = 28
    ROLL_CORR = 29
    ROLL_COV = 30
    ROLL_BETA = 31


BUILTIN_OPCODES: dict[str, Opcode] = {
    "vadd": Opcode.ADD,
    "vsub": Opcode.SUB,
    "vmul": Opcode.MUL,
    "vdiv": Opcode.DIV,
    "vneg": Opcode.NEG,
    "vabs": Opcode.ABS,
    "vlog": Opcode.LOG,
    "vsqrt": Opcode.SQRT,
    "vsin": Opcode.SIN,
    "vcos": Opcode.COS,
    "vgt": Opcode.GT,
    "vlt": Opcode.LT,
    "vge": Opcode.GE,
    "vle": Opcode.LE,
    "veq": Opcode.EQ,
    "vand": Opcode.AND,
    "vor": Opcode.OR,
    "vnot": Opcode.NOT,
    "vwhere": Opcode.WHERE,
    "delay": Opcode.DELAY,
    "diff": Opcode.DIFF,
    "rolling_sum": Opcode.ROLL_SUM,
    "rolling_mean": Opcode.ROLL_MEAN,
    "rolling_std": Opcode.ROLL_STD,
    "rolling_min": Opcode.ROLL_MIN,
    "rolling_max": Opcode.ROLL_MAX,
    "ema": Opcode.EMA,
    "rolling_corr": Opcode.ROLL_CORR,
    "rolling_cov": Opcode.ROLL_COV,
    "rolling_beta": Opcode.ROLL_BETA,
}
"""Opcode of every primitive registered by the builtin kits."""

OPCODES_ARITY: dict[int, int] = {
    Opcode.NEG: 1,
    Opcode.ABS: 1,
    Opcode.LOG: 1,
    Opcode.SQRT: 1,
    Opcode.SIN: 1,
    Opcode.COS: 1,
    Opcode.NOT: 1,
    Opcode.DELAY: 1,
    Opcode.DIFF: 1,
    Opcode.ROLL_SUM: 1,
    Opcode.ROLL_MEAN: 1,
    Opcode.ROLL_STD: 1,
    Opcode.ROLL_MIN: 1,
    Opcode.ROLL_MAX: 1,
    Opcode.EMA: 1,
    Opcode.ADD: 2,
    Opcode.SUB: 2,
    Opcode.MUL: 2,
    Opcode.DIV: 2,
    Opcode.GT: 2,
    Opcode.LT: 2,
    Opcode.GE: 2,
    Opcode.LE: 2,
    Opcode.EQ: 2,
    Opcode.AND: 2,
    Opcode.OR: 2,
    Opcode.ROLL_CORR: 2,
    Opcode.ROLL_COV: 2,
    Opcode.ROLL_BETA: 2,
    Opcode.WHERE: 3,
}

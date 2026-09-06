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
from typing import Any

from . import numba_codes as codes
from .numba_numeric import apply_numeric
from .numba_predicate import apply_predicate
from .numba_window import apply_window
from .numba_window_pair import apply_pair_window
from .numba_window_ts import apply_ts_window

__all__: list[str] = ["interpret", "idle"]


def interpret(  # pragma: no cover
    opcodes: Any,
    operands: Any,
    constants: Any,
    columns: Any,
    fill: float,
    stack: Any,
    scratch: Any,
    dispatch: Any,
) -> int:
    """Run one tape through the compiled stack machine.

    Args:
        opcodes: Instruction stream.
        operands: Immediate operand of each instruction.
        constants: Constant pool.
        columns: Packed input matrix.
        fill: Protected-op fill.
        stack: Column-length workspace.
        scratch: Spare row.
        dispatch: Consumer kernel.

    Returns:
        The stack pointer after the last instruction.

    Raises:
        ValueError: If an opcode is unknown.
    """
    rows = columns.shape[0]
    sp = 0
    for step in range(opcodes.size):
        op = opcodes[step]
        arg = operands[step]
        if op <= codes.COS:
            sp = apply_numeric(op, rows, sp, stack, columns, constants, arg, fill)
        elif op <= codes.WHERE:
            sp = apply_predicate(op, rows, sp, stack)
        elif op <= codes.EMA:
            sp = apply_window(op, rows, sp, stack, scratch, arg)
        elif op <= codes.ROLL_BETA:
            sp = apply_pair_window(op, rows, sp, stack, scratch, arg)
        elif op <= codes.TS_ARGMIN:
            sp = apply_ts_window(op, rows, sp, stack, scratch, arg)
        elif op >= codes.BASE:
            sp = int(dispatch(op, sp, stack, columns, constants, scratch))
        else:
            raise ValueError(codes.UNKNOWN_OPCODE)
    return sp


def idle(  # pragma: no cover
    _op: int,
    _sp: int,
    _stack: Any,
    _columns: Any,
    _constants: Any,
    _scratch: Any,
) -> int:
    """Reject an unexpected consumer opcode.

    The parameter names are unused; they exist so the fallback kernel
    matches ``USER_DISPATCH_SIGNATURE``.

    Returns:
        ``-1``, which is not a valid stack pointer.
    """
    return -1

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

__all__: list[str] = ["interpret", "idle", "interpret_many"]


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


def interpret_many(  # pragma: no cover
    run: Any,
    opcodes: Any,
    operands: Any,
    constants: Any,
    op_starts: Any,
    op_lens: Any,
    c_starts: Any,
    c_lens: Any,
    fills: Any,
    columns: Any,
    stack: Any,
    scratch: Any,
    dispatch: Any,
    out: Any,
) -> None:
    """Run many tapes through one compiled interpreter.

    Args:
        run: Compiled single-tape interpreter.
        opcodes: Concatenated instruction stream.
        operands: Concatenated immediates.
        constants: Concatenated constant pools.
        op_starts: Start index of each tape in ``opcodes``.
        op_lens: Instruction count of each tape.
        c_starts: Start index of each tape in ``constants``.
        c_lens: Constant-pool length of each tape.
        fills: Protected-op fill of each tape.
        columns: Packed input matrix.
        stack: Shared column-length workspace.
        scratch: Spare row.
        dispatch: Consumer kernel.
        out: Result of shape ``(n_tapes, n_rows)``.
    """
    rows = columns.shape[0]
    for index in range(op_starts.shape[0]):
        start = op_starts[index]
        n_ops = op_lens[index]
        const_start = c_starts[index]
        n_consts = c_lens[index]
        run(
            opcodes[start : start + n_ops],
            operands[start : start + n_ops],
            constants[const_start : const_start + n_consts],
            columns,
            fills[index],
            stack,
            scratch,
            dispatch,
        )
        for row in range(rows):
            out[index, row] = stack[0, row]

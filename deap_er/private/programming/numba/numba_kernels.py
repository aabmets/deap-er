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

__all__: list[str] = [
    "apply_builtin",
    "interpret",
    "interpret_dispatch",
    "tape_slices",
    "interpret_builtins",
    "interpret_many",
]


def apply_builtin(  # pragma: no cover
    op: int,
    arg: int,
    sp: int,
    stack: Any,
    scratch: Any,
    columns: Any,
    constants: Any,
    fill: float,
) -> int:
    """Run one builtin instruction.

    Args:
        op: Builtin opcode.
        arg: Immediate operand of the instruction.
        sp: Current stack pointer.
        stack: Column-length workspace.
        scratch: Spare row.
        columns: Packed input matrix.
        constants: Constant pool.
        fill: Protected-op fill.

    Returns:
        The stack pointer after the instruction.

    Raises:
        ValueError: If the opcode is not a builtin one.
    """
    rows = columns.shape[0]
    if op <= codes.COS:
        return apply_numeric(op, rows, sp, stack, columns, constants, arg, fill)
    if op <= codes.WHERE:
        return apply_predicate(op, rows, sp, stack)
    if op <= codes.EMA:
        return apply_window(op, rows, sp, stack, scratch, arg)
    if op <= codes.ROLL_BETA:
        return apply_pair_window(op, rows, sp, stack, scratch, arg)
    if op <= codes.TS_ARGMIN:
        return apply_ts_window(op, rows, sp, stack, scratch, arg)
    raise ValueError(codes.UNKNOWN_OPCODE)


def interpret(  # pragma: no cover
    opcodes: Any,
    operands: Any,
    constants: Any,
    columns: Any,
    fill: float,
    stack: Any,
    scratch: Any,
) -> int:
    """Run one tape of builtin instructions through the stack machine.

    It takes no consumer kernel, so its compiled form has the same
    type signature in every process and loads from the disk cache.

    Args:
        opcodes: Instruction stream.
        operands: Immediate operand of each instruction.
        constants: Constant pool.
        columns: Packed input matrix.
        fill: Protected-op fill.
        stack: Column-length workspace.
        scratch: Spare row.

    Returns:
        The stack pointer after the last instruction.

    Raises:
        ValueError: If an opcode is unknown or a consumer opcode.
    """
    sp = 0
    for step in range(opcodes.size):
        sp = apply_builtin(
            opcodes[step], operands[step], sp, stack, scratch, columns, constants, fill
        )
    return sp


def interpret_dispatch(  # pragma: no cover
    opcodes: Any,
    operands: Any,
    constants: Any,
    columns: Any,
    fill: float,
    stack: Any,
    scratch: Any,
    dispatch: Any,
) -> int:
    """Run one tape through the stack machine and a consumer kernel.

    Args:
        opcodes: Instruction stream.
        operands: Immediate operand of each instruction.
        constants: Constant pool.
        columns: Packed input matrix.
        fill: Protected-op fill.
        stack: Column-length workspace.
        scratch: Spare row.
        dispatch: Consumer kernel for the opcodes at or above ``BASE``.

    Returns:
        The stack pointer after the last instruction.

    Raises:
        ValueError: If an opcode is unknown.
    """
    sp = 0
    for step in range(opcodes.size):
        op = opcodes[step]
        if op >= codes.BASE:
            sp = int(dispatch(op, sp, stack, columns, constants, scratch))
        else:
            sp = apply_builtin(op, operands[step], sp, stack, scratch, columns, constants, fill)
    return sp


def tape_slices(
    index: int, streams: Any, layout: Any
) -> tuple[Any, Any, Any, float]:  # pragma: no cover
    """Cut one tape out of a packed batch.

    Args:
        index: Tape index.
        streams: ``(opcodes, operands, constants)`` concatenated tapes.
        layout: ``(op_starts, op_lens, c_starts, c_lens, fills)``.

    Returns:
        The tape's opcodes, operands, constants, and fill.
    """
    opcodes, operands, constants = streams
    op_starts, op_lens, c_starts, c_lens, fills = layout
    start = op_starts[index]
    stop = start + op_lens[index]
    const_start = c_starts[index]
    const_stop = const_start + c_lens[index]
    return (
        opcodes[start:stop],
        operands[start:stop],
        constants[const_start:const_stop],
        fills[index],
    )


def interpret_builtins(  # pragma: no cover
    streams: Any, layout: Any, columns: Any, stack: Any, scratch: Any, out: Any
) -> None:
    """Run many builtin-only tapes through the cached interpreter.

    It calls the interpreter as a global rather than taking it as an
    argument, which is what lets Numba cache it on disk.

    Args:
        streams: ``(opcodes, operands, constants)`` concatenated tapes.
        layout: ``(op_starts, op_lens, c_starts, c_lens, fills)``.
        columns: Packed input matrix.
        stack: Column-length workspace of this call.
        scratch: Spare row.
        out: Result of shape ``(n_tapes, n_rows)``.
    """
    for index in range(out.shape[0]):
        opcodes, operands, constants, fill = tape_slices(index, streams, layout)
        interpret(opcodes, operands, constants, columns, fill, stack, scratch)
        out[index] = stack[0]


def interpret_many(  # pragma: no cover
    run: Any, streams: Any, layout: Any, columns: Any, stack: Any, scratch: Any, out: Any
) -> None:
    """Run many tapes through one compiled interpreter.

    Args:
        run: Compiled single-tape interpreter.
        streams: ``(opcodes, operands, constants)`` concatenated tapes.
        layout: ``(op_starts, op_lens, c_starts, c_lens, fills)``.
        columns: Packed input matrix.
        stack: Column-length workspace of this call.
        scratch: Spare row.
        out: Result of shape ``(n_tapes, n_rows)``.
    """
    for index in range(out.shape[0]):
        opcodes, operands, constants, fill = tape_slices(index, streams, layout)
        run(opcodes, operands, constants, columns, fill, stack, scratch)
        out[index] = stack[0]

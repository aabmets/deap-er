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
import math
from typing import Any

from . import numba_codes as codes

__all__: list[str] = [
    "truthy",
    "apply_numeric",
    "apply_load",
    "apply_arith",
    "apply_unary",
    "apply_div",
    "apply_log",
    "apply_sqrt",
]


def truthy(value: float) -> bool:  # pragma: no cover
    """Treat a 0/1 mask cell as a boolean without an equality test.

    Args:
        value: Stack cell written by a comparison or logic opcode.

    Returns:
        True when ``value`` is away from zero.
    """
    return abs(value) > 0.0


def apply_numeric(  # pragma: no cover
    op: int,
    rows: int,
    sp: int,
    stack: Any,
    columns: Any,
    constants: Any,
    arg: int,
    fill: float,
) -> int:
    """Apply a load or arithmetic opcode.

    Args:
        op: Opcode in the numeric group.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        columns: Packed input matrix.
        constants: Constant pool.
        arg: Immediate operand.
        fill: Protected-op fill.

    Returns:
        The updated stack pointer.

    Raises:
        ValueError: If ``op`` is not a numeric opcode.
    """
    if op in (codes.COL_LOAD, codes.CONST):
        return apply_load(op, rows, sp, stack, columns, constants, arg)
    if op <= codes.DIV:
        return apply_arith(op, rows, sp, stack, fill)
    if op in (codes.NEG, codes.ABS, codes.SIN, codes.COS):
        return apply_unary(op, rows, sp, stack)
    if op == codes.LOG:
        return apply_log(rows, sp, stack, fill)
    if op == codes.SQRT:
        return apply_sqrt(rows, sp, stack, fill)
    raise ValueError(codes.UNKNOWN_OPCODE)


def apply_load(  # pragma: no cover
    op: int,
    rows: int,
    sp: int,
    stack: Any,
    columns: Any,
    constants: Any,
    arg: int,
) -> int:
    """Push a column or a constant onto the stack.

    Args:
        op: ``COL_LOAD`` or ``CONST``.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        columns: Packed input matrix.
        constants: Constant pool.
        arg: Column index or constant-pool index.

    Returns:
        The updated stack pointer.
    """
    if op == codes.COL_LOAD:
        for t in range(rows):
            stack[sp, t] = columns[t, arg]
        return sp + 1
    value = constants[arg]
    for t in range(rows):
        stack[sp, t] = value
    return sp + 1


def apply_arith(op: int, rows: int, sp: int, stack: Any, fill: float) -> int:  # pragma: no cover
    """Apply a binary arithmetic opcode.

    Args:
        op: ``ADD``, ``SUB``, ``MUL``, or ``DIV``.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        fill: Protected-op fill.

    Returns:
        The updated stack pointer.

    Raises:
        ValueError: If ``op`` is not a binary arithmetic opcode.
    """
    if op == codes.ADD:
        sp -= 1
        for t in range(rows):
            stack[sp - 1, t] = stack[sp - 1, t] + stack[sp, t]
        return sp
    if op == codes.SUB:
        sp -= 1
        for t in range(rows):
            stack[sp - 1, t] = stack[sp - 1, t] - stack[sp, t]
        return sp
    if op == codes.MUL:
        sp -= 1
        for t in range(rows):
            stack[sp - 1, t] = stack[sp - 1, t] * stack[sp, t]
        return sp
    if op == codes.DIV:
        return apply_div(rows, sp, stack, fill)
    raise ValueError(codes.UNKNOWN_OPCODE)


def apply_unary(op: int, rows: int, sp: int, stack: Any) -> int:  # pragma: no cover
    """Apply a unary numeric opcode.

    Args:
        op: ``NEG``, ``ABS``, ``SIN``, or ``COS``.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.

    Returns:
        The updated stack pointer.

    Raises:
        ValueError: If ``op`` is not a unary numeric opcode.
    """
    if op == codes.NEG:
        for t in range(rows):
            stack[sp - 1, t] = -stack[sp - 1, t]
        return sp
    if op == codes.ABS:
        for t in range(rows):
            stack[sp - 1, t] = abs(stack[sp - 1, t])
        return sp
    if op == codes.SIN:
        for t in range(rows):
            stack[sp - 1, t] = math.sin(stack[sp - 1, t])
        return sp
    if op == codes.COS:
        for t in range(rows):
            stack[sp - 1, t] = math.cos(stack[sp - 1, t])
        return sp
    raise ValueError(codes.UNKNOWN_OPCODE)


def apply_div(rows: int, sp: int, stack: Any, fill: float) -> int:  # pragma: no cover
    """Apply protected division.

    Args:
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        fill: Value used when finite operands produce a non-finite result.

    Returns:
        The updated stack pointer.
    """
    sp -= 1
    for t in range(rows):
        left = stack[sp - 1, t]
        right = stack[sp, t]
        value = left / right
        if not math.isfinite(value) and math.isfinite(left) and math.isfinite(right):
            value = fill
        stack[sp - 1, t] = value
    return sp


def apply_log(rows: int, sp: int, stack: Any, fill: float) -> int:  # pragma: no cover
    """Apply the protected natural logarithm.

    Args:
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        fill: Value used for a non-positive finite operand.

    Returns:
        The updated stack pointer.
    """
    for t in range(rows):
        value = stack[sp - 1, t]
        if value > 0.0:
            stack[sp - 1, t] = math.log(value)
        elif math.isfinite(value):
            stack[sp - 1, t] = fill
        else:
            stack[sp - 1, t] = math.nan
    return sp


def apply_sqrt(rows: int, sp: int, stack: Any, fill: float) -> int:  # pragma: no cover
    """Apply the protected square root.

    Args:
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        fill: Value used for a negative finite operand.

    Returns:
        The updated stack pointer.
    """
    for t in range(rows):
        value = stack[sp - 1, t]
        if value >= 0.0:
            stack[sp - 1, t] = math.sqrt(value)
        elif math.isfinite(value):
            stack[sp - 1, t] = fill
        else:
            stack[sp - 1, t] = math.nan
    return sp

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

from ._numba_codes import _AND, _EQ, _GE, _GT, _LE, _LT, _NOT, _OR, _UNKNOWN_OPCODE, _WHERE
from ._numba_numeric import _truthy

__all__: list[str] = []


def _apply_predicate(op: int, rows: int, sp: int, stack: Any) -> int:  # pragma: no cover
    """Apply a comparison, logic, or selection opcode.

    Args:
        op: Opcode in the predicate group.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.

    Returns:
        The updated stack pointer.

    Raises:
        ValueError: If ``op`` is not a predicate opcode.
    """
    if op in (_GT, _LT):
        return _apply_gt_lt(op, rows, sp, stack)
    if op in (_GE, _LE):
        return _apply_ge_le(op, rows, sp, stack)
    if op == _EQ:
        return _apply_eq(rows, sp, stack)
    if op in (_AND, _OR):
        return _apply_and_or(op, rows, sp, stack)
    if op <= _WHERE:
        return _apply_not_where(op, rows, sp, stack)
    raise ValueError(_UNKNOWN_OPCODE)


def _apply_gt_lt(op: int, rows: int, sp: int, stack: Any) -> int:  # pragma: no cover
    """Apply a strict comparison opcode.

    Args:
        op: ``GT`` or ``LT``.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.

    Returns:
        The updated stack pointer.
    """
    sp -= 1
    if op == _GT:
        for t in range(rows):
            stack[sp - 1, t] = 1.0 if stack[sp - 1, t] > stack[sp, t] else 0.0
        return sp
    for t in range(rows):
        stack[sp - 1, t] = 1.0 if stack[sp - 1, t] < stack[sp, t] else 0.0
    return sp


def _apply_ge_le(op: int, rows: int, sp: int, stack: Any) -> int:  # pragma: no cover
    """Apply an inclusive comparison opcode.

    Args:
        op: ``GE`` or ``LE``.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.

    Returns:
        The updated stack pointer.
    """
    sp -= 1
    if op == _GE:
        for t in range(rows):
            stack[sp - 1, t] = 1.0 if stack[sp - 1, t] >= stack[sp, t] else 0.0
        return sp
    for t in range(rows):
        stack[sp - 1, t] = 1.0 if stack[sp - 1, t] <= stack[sp, t] else 0.0
    return sp


def _apply_eq(rows: int, sp: int, stack: Any) -> int:  # pragma: no cover
    """Apply exact equality without a floating-point ``==``.

    Both inequalities are True only when the values are equal and
    finite, matching ``numpy.equal``.

    Args:
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.

    Returns:
        The updated stack pointer.
    """
    sp -= 1
    for t in range(rows):
        left = stack[sp - 1, t]
        right = stack[sp, t]
        stack[sp - 1, t] = 1.0 if left <= right and left >= right else 0.0
    return sp


def _apply_and_or(op: int, rows: int, sp: int, stack: Any) -> int:  # pragma: no cover
    """Apply a binary logic opcode.

    Args:
        op: ``AND`` or ``OR``.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.

    Returns:
        The updated stack pointer.
    """
    sp -= 1
    if op == _AND:
        for t in range(rows):
            stack[sp - 1, t] = 1.0 if _truthy(stack[sp - 1, t]) and _truthy(stack[sp, t]) else 0.0
        return sp
    for t in range(rows):
        stack[sp - 1, t] = 1.0 if _truthy(stack[sp - 1, t]) or _truthy(stack[sp, t]) else 0.0
    return sp


def _apply_not_where(op: int, rows: int, sp: int, stack: Any) -> int:  # pragma: no cover
    """Apply ``NOT`` or ``WHERE``.

    Args:
        op: ``NOT`` or ``WHERE``.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.

    Returns:
        The updated stack pointer.
    """
    if op == _NOT:
        for t in range(rows):
            stack[sp - 1, t] = 0.0 if _truthy(stack[sp - 1, t]) else 1.0
        return sp
    sp -= 2
    for t in range(rows):
        stack[sp - 1, t] = stack[sp, t] if _truthy(stack[sp - 1, t]) else stack[sp + 1, t]
    return sp

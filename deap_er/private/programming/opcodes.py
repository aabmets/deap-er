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
from collections.abc import Callable, Sequence
from typing import Any

from . import window_ops, window_pair
from .numpy import numpy_ops
from .opcode_set import BUILTIN_OPCODES, USER_BASE, Opcode
from .tape import Tape, bind_numba_opcode, numba_opcodes
from .tape_lower import lower_tree

__all__: list[str] = [
    "Opcode",
    "USER_BASE",
    "Tape",
    "BUILTIN_OPCODES",
    "bind_numba_opcode",
    "numba_opcodes",
    "lower_tree",
    "interpret_tape",
]

_STACK_UNDERFLOW = "The tape is malformed and underflows the evaluation stack."

_UNARY: dict[int, Callable[..., Any]] = {
    Opcode.NEG: numpy_ops.vneg,
    Opcode.ABS: numpy_ops.vabs,
    Opcode.SIN: numpy_ops.vsin,
    Opcode.COS: numpy_ops.vcos,
    Opcode.NOT: numpy_ops.vnot,
}
_UNARY_PROTECTED: dict[int, Callable[..., Any]] = {
    Opcode.LOG: numpy_ops.vlog,
    Opcode.SQRT: numpy_ops.vsqrt,
}
_BINARY: dict[int, Callable[..., Any]] = {
    Opcode.ADD: numpy_ops.vadd,
    Opcode.SUB: numpy_ops.vsub,
    Opcode.MUL: numpy_ops.vmul,
    Opcode.GT: numpy_ops.vgt,
    Opcode.LT: numpy_ops.vlt,
    Opcode.GE: numpy_ops.vge,
    Opcode.LE: numpy_ops.vle,
    Opcode.EQ: numpy_ops.veq,
    Opcode.AND: numpy_ops.vand,
    Opcode.OR: numpy_ops.vor,
}
_BINARY_PROTECTED: dict[int, Callable[..., Any]] = {Opcode.DIV: numpy_ops.vdiv}
_WINDOWED: dict[int, Callable[..., Any]] = {
    Opcode.DELAY: window_ops.delay,
    Opcode.DIFF: window_ops.diff,
    Opcode.ROLL_SUM: window_ops.rolling_sum,
    Opcode.ROLL_MEAN: window_ops.rolling_mean,
    Opcode.ROLL_STD: window_ops.rolling_std,
    Opcode.ROLL_MIN: window_ops.rolling_min,
    Opcode.ROLL_MAX: window_ops.rolling_max,
    Opcode.EMA: window_ops.ema,
}
_PAIR_WINDOWED: dict[int, Callable[..., Any]] = {
    Opcode.ROLL_CORR: window_pair.rolling_corr,
    Opcode.ROLL_COV: window_pair.rolling_cov,
    Opcode.ROLL_BETA: window_pair.rolling_beta,
}


def interpret_tape(tape: Tape, columns: Sequence[Any]) -> Any:
    """Run a tape over a sequence of columns.

    Evaluates through the same functions as the default backend, so
    the two agree by construction.

    Args:
        tape: Tape produced by ``lower_tree``.
        columns: One array per column, in the order the primitive set
            declares them.

    Returns:
        The result of the expression.

    Raises:
        ValueError: If the column count does not match the tape, if
            the tape holds an instruction the interpreter does not
            know, or if the tape underflows or leaves no result.
    """
    if len(columns) != tape.columns:
        raise ValueError(f"The tape expects {tape.columns} columns, got {len(columns)}.")

    stack: list[Any] = []
    for step in range(tape.opcodes.size):
        opcode = int(tape.opcodes[step])
        operand = int(tape.operands[step])
        _apply_opcode(stack, columns, tape, opcode, operand)
    if not stack:
        raise ValueError("The tape is malformed and leaves no result.")
    return stack[-1]


def _peek(stack: list[Any]) -> Any:
    """Return the top stack value, or raise if the tape underflowed.

    Args:
        stack: Evaluation stack.

    Returns:
        The current top of the stack.

    Raises:
        ValueError: If the stack is empty.
    """
    if not stack:
        raise ValueError(_STACK_UNDERFLOW)
    return stack[-1]


def _replace(stack: list[Any], value: Any) -> None:
    """Overwrite the top stack value, or raise if the tape underflowed.

    Args:
        stack: Evaluation stack.
        value: Replacement for the current top.

    Raises:
        ValueError: If the stack is empty.
    """
    if not stack:
        raise ValueError(_STACK_UNDERFLOW)
    stack[-1] = value


def _pop(stack: list[Any]) -> Any:
    """Pop the top stack value, or raise if the tape underflowed.

    Args:
        stack: Evaluation stack.

    Returns:
        The previous top of the stack.

    Raises:
        ValueError: If the stack is empty.
    """
    if not stack:
        raise ValueError(_STACK_UNDERFLOW)
    return stack.pop()


def _apply_opcode(
    stack: list[Any], columns: Sequence[Any], tape: Tape, opcode: int, operand: int
) -> None:
    """Apply one tape instruction to the evaluation stack.

    Args:
        stack: Evaluation stack.
        columns: Column arrays aligned with the tape.
        tape: Tape that produced ``opcode``.
        opcode: Instruction to apply.
        operand: Immediate operand of the instruction.

    Raises:
        ValueError: If the instruction is unknown or the stack
            underflows.
    """
    if opcode == Opcode.COL_LOAD:
        stack.append(columns[operand])
        return
    if opcode == Opcode.CONST:
        stack.append(float(tape.constants[operand]))
        return
    if opcode in _UNARY:
        _replace(stack, _UNARY[opcode](_peek(stack)))
        return
    if opcode in _UNARY_PROTECTED:
        _replace(stack, _UNARY_PROTECTED[opcode](_peek(stack), fill=tape.fill))
        return
    if opcode in _WINDOWED:
        _replace(stack, _WINDOWED[opcode](_peek(stack), operand))
        return
    if opcode in _PAIR_WINDOWED:
        right = _pop(stack)
        _replace(stack, _PAIR_WINDOWED[opcode](_peek(stack), right, operand))
        return
    if opcode in _BINARY:
        right = _pop(stack)
        _replace(stack, _BINARY[opcode](_peek(stack), right))
        return
    if opcode in _BINARY_PROTECTED:
        right = _pop(stack)
        _replace(stack, _BINARY_PROTECTED[opcode](_peek(stack), right, fill=tape.fill))
        return
    if opcode == Opcode.WHERE:
        on_false = _pop(stack)
        on_true = _pop(stack)
        _replace(stack, numpy_ops.vwhere(_peek(stack), on_true, on_false))
        return
    raise ValueError(
        f"Opcode {opcode} has no Python implementation. Consumer opcodes "
        f"are only available on the Numba backend."
    )

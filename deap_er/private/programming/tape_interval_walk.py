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

import numpy

from .opcode_set import OPCODES_ARITY, USER_BASE, Opcode
from .tape import Tape
from .tape_interval_ops import (
    BINARY,
    BINARY_PROTECTED,
    COMPARISONS,
    PAIR_WINDOWED,
    STACK_UNDERFLOW,
    UNARY,
    WINDOWED,
    Summary,
    apply_binary,
    apply_unary,
    hides_warmup,
    merge_arrays,
)
from .tape_interval_window import apply_pair_window, apply_window

__all__: list[str] = ["WalkResult", "walk_tape"]

_CONSUMER_INTERVAL = (
    "Consumer opcode {opcode} has no interval certificate. "
    "Rescore the full matrix with interpret_tapes."
)


class WalkResult:
    """Postfix walk output for interval analysis."""

    __slots__ = ("summary", "warmup_hidden")

    def __init__(self, summary: Summary, warmup_hidden: bool) -> None:
        """Store the root summary and warmup flag."""
        self.summary = summary
        self.warmup_hidden = warmup_hidden


def walk_tape(tape: Tape, column_bounds: numpy.ndarray) -> WalkResult:
    """Walk a tape and return the root summary plus warmup flags."""
    stack: list[Summary] = []
    hides = False
    for step in range(tape.opcodes.size):
        hides |= _apply_step(stack, tape, step, column_bounds)
    if not stack:
        raise ValueError("The tape is malformed and leaves no result.")
    root = stack[-1]
    if root.kind != "array":
        raise ValueError("The tape is malformed and leaves no result.")
    return WalkResult(root, hides)


def _apply_step(stack: list[Summary], tape: Tape, step: int, column_bounds: numpy.ndarray) -> bool:
    opcode = int(tape.opcodes[step])
    operand = int(tape.operands[step])
    if opcode >= USER_BASE:
        raise ValueError(_CONSUMER_INTERVAL.format(opcode=opcode))
    if opcode == int(Opcode.COL_LOAD):
        _push_col_load(stack, column_bounds, operand)
        return False
    if opcode == int(Opcode.CONST):
        _push_const(stack, tape, operand)
        return False
    if opcode in COMPARISONS:
        _push_comparison(stack)
        return False
    if opcode == int(Opcode.NOT):
        _push_mask_not(stack)
        return False
    if opcode in {int(Opcode.AND), int(Opcode.OR)}:
        _push_mask_combine(stack)
        return False
    if opcode == int(Opcode.WHERE):
        return _push_where(stack)
    if opcode in UNARY:
        _push_unary(stack, tape, opcode)
        return False
    if opcode in BINARY or opcode in BINARY_PROTECTED:
        _push_binary(stack, tape, opcode)
        return False
    if opcode in WINDOWED:
        _push_window(stack, opcode, operand)
        return False
    if opcode in PAIR_WINDOWED:
        _push_pair_window(stack, opcode, operand)
        return False
    if opcode not in OPCODES_ARITY:
        raise ValueError(f"Opcode {opcode} has no interval certificate.")
    raise ValueError(f"Opcode {opcode} has no interval certificate.")


def _push_col_load(stack: list[Summary], column_bounds: numpy.ndarray, operand: int) -> None:
    lo, hi = float(column_bounds[operand, 0]), float(column_bounds[operand, 1])
    can_finite = numpy.isfinite(lo) or numpy.isfinite(hi)
    stack.append(Summary(lo, hi, 0, 0, lo == hi, can_finite, "array"))


def _push_const(stack: list[Summary], tape: Tape, operand: int) -> None:
    value = float(tape.constants[operand])
    stack.append(Summary(value, value, 0, 0, True, numpy.isfinite(value), "array"))


def _push_comparison(stack: list[Summary]) -> None:
    right = _pop_array(stack)
    left = _pop_array(stack)
    compared = max(left.lookback, right.lookback)
    stack.append(Summary(0.0, 1.0, 0, 0, False, True, "mask", compared))


def _push_mask_not(stack: list[Summary]) -> None:
    mask = _pop_mask(stack)
    stack.append(Summary(0.0, 1.0, 0, 0, False, True, "mask", mask.compared_lookback))


def _push_mask_combine(stack: list[Summary]) -> None:
    right = _pop_mask(stack)
    left = _pop_mask(stack)
    stack.append(
        Summary(
            0.0,
            1.0,
            0,
            0,
            False,
            True,
            "mask",
            max(left.compared_lookback, right.compared_lookback),
        )
    )


def _push_where(stack: list[Summary]) -> bool:
    on_false = _pop_array(stack)
    on_true = _pop_array(stack)
    condition = _pop_mask(stack)
    stack.append(merge_arrays(on_true, on_false))
    return hides_warmup(condition, on_true, on_false)


def _push_unary(stack: list[Summary], tape: Tape, opcode: int) -> None:
    child = _pop_array(stack)
    stack.append(apply_unary(opcode, child, tape.fill))


def _push_binary(stack: list[Summary], tape: Tape, opcode: int) -> None:
    right = _pop_array(stack)
    left = _pop_array(stack)
    stack.append(apply_binary(opcode, left, right, tape.fill))


def _push_window(stack: list[Summary], opcode: int, operand: int) -> None:
    child = _pop_array(stack)
    stack.append(apply_window(opcode, child, operand))


def _push_pair_window(stack: list[Summary], opcode: int, operand: int) -> None:
    right = _pop_array(stack)
    left = _pop_array(stack)
    stack.append(apply_pair_window(opcode, left, right, operand))


def _pop_array(stack: list[Summary]) -> Summary:
    value = _pop(stack)
    if value.kind != "array":
        raise ValueError(STACK_UNDERFLOW)
    return value


def _pop_mask(stack: list[Summary]) -> Summary:
    value = _pop(stack)
    if value.kind == "mask":
        return value
    if value.kind != "array":
        raise ValueError(STACK_UNDERFLOW)
    return Summary(0.0, 1.0, 0, 0, False, True, "mask", value.lookback)


def _pop(stack: list[Summary]) -> Summary:
    if not stack:
        raise ValueError(STACK_UNDERFLOW)
    return stack.pop()

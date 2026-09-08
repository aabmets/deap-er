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
from .opcode_set import OPCODES_ARITY, USER_BASE, Opcode
from .tape import Tape

__all__: list[str] = ["opcode_lookback", "tape_lookback"]

_STACK_UNDERFLOW = "The tape is malformed and underflows the lookback stack."
_CONSUMER_LOOKBACK = (
    "Consumer opcode {opcode} has no lookback certificate. "
    "Rescore the full matrix with interpret_tapes."
)

_DELAY_STEPS = frozenset({int(Opcode.DELAY), int(Opcode.DIFF)})
_EMA_WARMUP = frozenset({int(Opcode.EMA)})
_WINDOW_ARG = frozenset(
    {
        int(Opcode.ROLL_SUM),
        int(Opcode.ROLL_MEAN),
        int(Opcode.ROLL_STD),
        int(Opcode.ROLL_MIN),
        int(Opcode.ROLL_MAX),
        int(Opcode.ROLL_CORR),
        int(Opcode.ROLL_COV),
        int(Opcode.ROLL_BETA),
        int(Opcode.TS_RANK),
        int(Opcode.TS_ARGMAX),
        int(Opcode.TS_ARGMIN),
    }
)


def opcode_lookback(opcode: int, operand: int) -> int:
    """Return the finite lookback one instruction adds.

    Delay and diff look back ``operand`` steps. Rolling, pair-window,
    and time-series opcodes use the ``Window`` argument. EMA uses the
    documented warmup of ``window - 1``. Pointwise opcodes add nothing.

    Args:
        opcode: Instruction to classify.
        operand: Immediate operand of the instruction.

    Returns:
        Rows of earlier history that instruction needs.

    Raises:
        ValueError: If ``opcode`` is a consumer kernel or is unknown.
    """
    if opcode in _DELAY_STEPS:
        return operand
    if opcode in _EMA_WARMUP:
        return max(operand - 1, 0)
    if opcode in _WINDOW_ARG:
        return operand
    if opcode in OPCODES_ARITY or opcode in {int(Opcode.COL_LOAD), int(Opcode.CONST)}:
        return 0
    if opcode >= USER_BASE:
        raise ValueError(_CONSUMER_LOOKBACK.format(opcode=opcode))
    raise ValueError(f"Opcode {opcode} has no lookback certificate.")


def tape_lookback(tape: Tape) -> int:
    """Return the causal lookback bound of a tape.

    Walks the postfix tape and composes per-opcode lookbacks: windowed
    instructions add their bound to the child, and pointwise
    instructions take the max of their arguments. The result is the
    shortest prefix of earlier rows that a legal suffix rescore must
    keep as history.

    Args:
        tape: Tape produced by ``lower_tree``.

    Returns:
        The program's lookback in rows. ``0`` for a pointwise tape.

    Raises:
        ValueError: If the tape underflows, leaves no result, or holds
            a consumer opcode.
    """
    stack: list[int] = []
    for step in range(tape.opcodes.size):
        opcode = int(tape.opcodes[step])
        extra = opcode_lookback(opcode, int(tape.operands[step]))
        arity = OPCODES_ARITY.get(opcode)
        if opcode in {int(Opcode.COL_LOAD), int(Opcode.CONST)}:
            stack.append(0)
            continue
        if arity == 1:
            stack[-1] = _peek(stack) + extra
            continue
        if arity == 2:
            right = _pop(stack)
            stack[-1] = max(_peek(stack), right) + extra
            continue
        if arity == 3:
            on_false = _pop(stack)
            on_true = _pop(stack)
            stack[-1] = max(_peek(stack), on_true, on_false)
            continue
        raise ValueError(f"Opcode {opcode} has no lookback certificate.")
    if not stack:
        raise ValueError("The tape is malformed and leaves no result.")
    return stack[-1]


def _peek(stack: list[int]) -> int:
    """Return the top lookback, or raise if the tape underflowed."""
    if not stack:
        raise ValueError(_STACK_UNDERFLOW)
    return stack[-1]


def _pop(stack: list[int]) -> int:
    """Pop the top lookback, or raise if the tape underflowed."""
    if not stack:
        raise ValueError(_STACK_UNDERFLOW)
    return stack.pop()

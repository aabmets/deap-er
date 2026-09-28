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
from dataclasses import dataclass

import numpy

from .opcode_set import BUILTIN_OPCODES, OPCODES_ARITY, USER_BASE, Opcode

__all__: list[str] = ["Tape", "bind_numba_opcode", "check_tape", "numba_opcodes", "opcode_of"]

_user_opcodes: dict[str, int] = {}
_WINDOW_OPCODES = range(int(Opcode.DELAY), int(Opcode.TS_ARGMIN) + 1)


@dataclass(frozen=True, eq=False)
class Tape:
    """Flat, picklable form of a compiled expression tree.

    Instructions are in postfix order. ``operands[i]`` is the column
    index of a ``COL_LOAD``, the constant pool index of a ``CONST``,
    the window length of a rolling instruction, and ``-1`` otherwise.

    Attributes:
        opcodes: Instruction stream as ``int32``.
        operands: Immediate operand of each instruction as ``int32``.
        constants: Constant pool as ``float64``.
        columns: Number of columns the tape expects.
        depth: Peak stack depth reached while running the tape.
        fill: Fill used by the protected instructions.
    """

    opcodes: numpy.ndarray
    operands: numpy.ndarray
    constants: numpy.ndarray
    columns: int
    depth: int
    fill: float


def bind_numba_opcode(name: str, opcode: int) -> None:
    """Bind a primitive name to a consumer-supplied opcode.

    The binding is used only while lowering a tree. It is never read
    inside the interpreter loop. Persist the bindings alongside a run:
    replaying a checkpointed tree against different bindings decodes
    the tree into different instructions.

    Args:
        name: Name of a primitive registered on the primitive set.
        opcode: Opcode value, at or above ``USER_BASE``.

    Raises:
        ValueError: If the opcode is below ``USER_BASE``, if the name
            belongs to a builtin primitive, or if the name is already
            bound to a different opcode.
    """
    if opcode < USER_BASE:
        raise ValueError(f"Consumer opcodes must be at least {USER_BASE}, got {opcode}.")
    if name in BUILTIN_OPCODES:
        raise ValueError(f"The primitive '{name}' already has a builtin opcode.")
    known = _user_opcodes.get(name)
    if known is not None and known != opcode:
        raise ValueError(f"The primitive '{name}' is already bound to opcode {known}.")
    _user_opcodes[name] = opcode


def numba_opcodes() -> dict[str, int]:
    """Return the consumer opcode bindings made so far.

    Returns:
        A copy of the name to opcode map, suitable for storing with a
        checkpoint.
    """
    return dict(_user_opcodes)


def opcode_of(name: str) -> int:
    """Resolve a primitive name to an opcode.

    Args:
        name: Name of the primitive.

    Returns:
        The builtin or bound opcode.

    Raises:
        ValueError: If the name has neither.
    """
    builtin = BUILTIN_OPCODES.get(name)
    if builtin is not None:
        return int(builtin)
    bound = _user_opcodes.get(name)
    if bound is not None:
        return bound
    raise ValueError(
        f"The primitive '{name}' has no builtin opcode and no binding. "
        f"Register one with bind_numba_opcode before lowering."
    )


def check_tape(tape: Tape) -> None:
    """Reject a tape that would index outside the compiled workspace.

    The compiled interpreter trusts the tape. One pass over the
    instructions checks what it would otherwise read or write out of
    bounds: column and constant operands, window lengths, stack
    underflow, and a peak depth above ``tape.depth``. A consumer
    opcode has no known arity, so stack accounting stops at the first
    one while the operand checks go on.

    Args:
        tape: Tape about to run on the compiled interpreter.

    Raises:
        ValueError: If the tape is malformed.
    """
    if tape.opcodes.shape != tape.operands.shape or tape.opcodes.ndim != 1:
        raise ValueError("The tape opcodes and operands must be aligned one-dimensional arrays.")
    pointer: int | None = 0
    for opcode, operand in zip(tape.opcodes.tolist(), tape.operands.tolist(), strict=True):
        _check_operand(tape, opcode, operand)
        if pointer is None:
            continue
        if opcode >= USER_BASE:
            pointer = None
            continue
        arity = OPCODES_ARITY.get(opcode, 0)
        if pointer < arity:
            raise ValueError("The tape is malformed and underflows the evaluation stack.")
        pointer += 1 - arity
        if pointer > tape.depth:
            raise ValueError(f"The tape reaches depth {pointer}, above its declared {tape.depth}.")
    if pointer is not None and pointer != 1:
        raise ValueError("The tape is malformed and does not leave exactly one result.")


def _check_operand(tape: Tape, opcode: int, operand: int) -> None:
    """Reject one out-of-range immediate operand."""
    if opcode == Opcode.COL_LOAD:
        valid = 0 <= operand < tape.columns
    elif opcode == Opcode.CONST:
        valid = 0 <= operand < tape.constants.size
    elif opcode in _WINDOW_OPCODES:
        valid = operand >= 1
    elif opcode in OPCODES_ARITY or opcode >= USER_BASE:
        valid = True
    else:
        raise ValueError(f"The tape holds opcode {opcode}, which is not an instruction.")
    if not valid:
        raise ValueError(f"The tape holds opcode {opcode} with an invalid operand {operand}.")

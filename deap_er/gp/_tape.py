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

from ._opcode_set import BUILTIN_OPCODES, USER_BASE

__all__: list[str] = []

_user_opcodes: dict[str, int] = {}


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


def _opcode_of(name: str) -> int:
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

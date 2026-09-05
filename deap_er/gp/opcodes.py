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
from dataclasses import dataclass
from enum import IntEnum
from numbers import Real
from typing import Any

import numpy

from . import numpy_ops, window_ops
from .columnar import Window
from .primitives import Primitive, PrimitiveSetTyped, PrimitiveTree
from .typedefs import GPExprTypes

__all__ = [
    "Opcode",
    "USER_BASE",
    "Tape",
    "BUILTIN_OPCODES",
    "bind_numba_opcode",
    "numba_opcodes",
    "lower_tree",
    "interpret_tape",
]

USER_BASE = 1000
"""First opcode value reserved for consumer-supplied kernels."""


class Opcode(IntEnum):
    """Instruction set of the tree stack machine.

    Every value below ``USER_BASE`` is owned by this library. Consumers
    bind their own kernels to values at or above ``USER_BASE``.
    """

    COL_LOAD = 0
    CONST = 1
    ADD = 2
    SUB = 3
    MUL = 4
    DIV = 5
    NEG = 6
    ABS = 7
    LOG = 8
    SQRT = 9
    SIN = 10
    COS = 11
    GT = 12
    LT = 13
    GE = 14
    LE = 15
    EQ = 16
    AND = 17
    OR = 18
    NOT = 19
    WHERE = 20
    DELAY = 21
    DIFF = 22
    ROLL_SUM = 23
    ROLL_MEAN = 24
    ROLL_STD = 25
    ROLL_MIN = 26
    ROLL_MAX = 27
    EMA = 28


BUILTIN_OPCODES: dict[str, Opcode] = {
    "vadd": Opcode.ADD,
    "vsub": Opcode.SUB,
    "vmul": Opcode.MUL,
    "vdiv": Opcode.DIV,
    "vneg": Opcode.NEG,
    "vabs": Opcode.ABS,
    "vlog": Opcode.LOG,
    "vsqrt": Opcode.SQRT,
    "vsin": Opcode.SIN,
    "vcos": Opcode.COS,
    "vgt": Opcode.GT,
    "vlt": Opcode.LT,
    "vge": Opcode.GE,
    "vle": Opcode.LE,
    "veq": Opcode.EQ,
    "vand": Opcode.AND,
    "vor": Opcode.OR,
    "vnot": Opcode.NOT,
    "vwhere": Opcode.WHERE,
    "delay": Opcode.DELAY,
    "diff": Opcode.DIFF,
    "rolling_sum": Opcode.ROLL_SUM,
    "rolling_mean": Opcode.ROLL_MEAN,
    "rolling_std": Opcode.ROLL_STD,
    "rolling_min": Opcode.ROLL_MIN,
    "rolling_max": Opcode.ROLL_MAX,
    "ema": Opcode.EMA,
}
"""Opcode of every primitive registered by the builtin kits."""

_user_opcodes: dict[str, int] = {}

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

_ARITY: dict[int, int] = {
    **dict.fromkeys(_UNARY, 1),
    **dict.fromkeys(_UNARY_PROTECTED, 1),
    **dict.fromkeys(_WINDOWED, 1),
    **dict.fromkeys(_BINARY, 2),
    **dict.fromkeys(_BINARY_PROTECTED, 2),
    Opcode.WHERE: 3,
}


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


def _child_indices(nodes: Sequence[Any]) -> list[list[int]]:
    """Map every node to its child indices, left to right.

    Args:
        nodes: Tree nodes in prefix order.

    Returns:
        A list of child index lists, aligned with ``nodes``.
    """
    children: list[list[int]] = [[] for _ in nodes]
    stack: list[list[int]] = []
    for index, node in enumerate(nodes):
        if stack:
            children[stack[-1][0]].append(index)
            stack[-1][1] -= 1
        stack.append([index, node.arity])
        while stack and stack[-1][1] == 0:
            stack.pop()
    return children


def _immediate_windows(
    nodes: Sequence[Any], children: list[list[int]]
) -> tuple[dict[int, int], set[int]]:
    """Fold the window argument of every builtin rolling node inline.

    Args:
        nodes: Tree nodes in prefix order.
        children: Child indices of every node.

    Returns:
        The window length of each rolling node, and the set of child
        indices that must not be pushed onto the stack.

    Raises:
        ValueError: If a window argument is not a leaf, or if its value
            is not an integer.
    """
    windows: dict[int, int] = {}
    folded: set[int] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, Primitive) or _opcode_of(node.name) >= USER_BASE:
            continue
        for position, arg_type in enumerate(node.args):
            if arg_type is not Window:
                continue
            child = children[index][position]
            value = getattr(nodes[child], "value", None)
            if nodes[child].arity != 0 or not isinstance(value, Real):
                raise ValueError(
                    f"The window argument of '{node.name}' must be a leaf holding "
                    f"an integer, so that it can be lowered to an immediate operand."
                )
            windows[index] = int(value)
            folded.add(child)
    return windows, folded


def _postfix_order(children: list[list[int]], folded: set[int]) -> list[int]:
    """Order node indices so that children come before their parent.

    Args:
        children: Child indices of every node.
        folded: Child indices that were folded into an immediate.

    Returns:
        Node indices in postfix order, left to right.
    """
    order: list[int] = []
    work: list[tuple[int, bool]] = [(0, False)]
    while work:
        index, expanded = work.pop()
        if expanded:
            order.append(index)
            continue
        work.append((index, True))
        for child in reversed(children[index]):
            if child not in folded:
                work.append((child, False))
    return order


def _leaf_instruction(
    node: Any, prim_set: PrimitiveSetTyped, pool: dict[float, int], constants: list[float]
) -> tuple[int, int]:
    """Lower a leaf node to a load instruction.

    A terminal is a column when its value names an argument of the
    set. Otherwise the value itself, or the object the name resolves
    to in the evaluation context, must be a number.

    Args:
        node: Terminal to lower.
        prim_set: Primitive set the tree was built from.
        pool: Constant value to pool index map.
        constants: Constant pool being filled.

    Returns:
        The opcode and its immediate operand.

    Raises:
        ValueError: If the terminal is neither a column nor a number.
    """
    value = getattr(node, "value", None)
    if isinstance(value, str):
        if value in prim_set.arguments:
            return int(Opcode.COL_LOAD), prim_set.arguments.index(value)
        value = prim_set.context.get(value)
    if isinstance(value, Real):
        number = float(value)
        index = pool.get(number)
        if index is None:
            index = len(constants)
            pool[number] = index
            constants.append(number)
        return int(Opcode.CONST), index
    raise ValueError(f"The terminal {node} cannot be lowered to an opcode.")


def lower_tree(
    expr: GPExprTypes, prim_set: PrimitiveSetTyped, *, fill: float | None = None
) -> Tape:
    """Lower an expression tree to a flat instruction tape.

    Only primitives with a builtin opcode or a binding made through
    ``bind_numba_opcode`` can be lowered. Everything else fails here
    rather than at evaluation time.

    Args:
        expr: Expression to lower. A ``PrimitiveTree``, a sequence of
            nodes, or a string that ``PrimitiveTree.from_string`` can
            parse against ``prim_set``.
        prim_set: Primitive set the expression was built from.
        fill: Fill for the protected instructions. Read back from
            ``prim_set`` when omitted.

    Returns:
        The lowered tape.

    Raises:
        ValueError: If a primitive has no opcode, if a window argument
            is not an integer leaf, if a terminal is neither a column
            nor a number, or if the tree is empty, holds unreachable
            nodes, or does not balance the evaluation stack.
    """
    if isinstance(expr, str):
        expr = PrimitiveTree.from_string(expr, prim_set)
    nodes = list(expr)
    if not nodes:
        raise ValueError("An empty expression cannot be lowered.")

    children = _child_indices(nodes)
    windows, folded = _immediate_windows(nodes, children)
    order = _postfix_order(children, folded)
    if len(order) + len(folded) != len(nodes):
        raise ValueError("The expression holds nodes that are not reachable from the root.")

    opcodes: list[int] = []
    operands: list[int] = []
    constants: list[float] = []
    pool: dict[float, int] = {}
    pointer = 0
    depth = 0

    for index in order:
        node = nodes[index]
        if not isinstance(node, Primitive):
            opcode, operand = _leaf_instruction(node, prim_set, pool, constants)
            pointer += 1
        else:
            opcode = _opcode_of(node.name)
            operand = windows.get(index, -1)
            # Builtins are all in _ARITY, and they are the only nodes
            # that fold an operand into the instruction. A consumer
            # kernel takes every argument from the stack.
            pointer -= _ARITY.get(opcode, node.arity) - 1
        if pointer < 1:
            raise ValueError("The expression is malformed and underflows the evaluation stack.")
        depth = max(depth, pointer)
        opcodes.append(opcode)
        operands.append(operand)

    if pointer != 1:
        raise ValueError("The expression is malformed and leaves more than one result.")

    return Tape(
        opcodes=numpy.array(opcodes, dtype=numpy.int32),
        operands=numpy.array(operands, dtype=numpy.int32),
        constants=numpy.array(constants, dtype=numpy.float64),
        columns=len(prim_set.arguments),
        depth=depth,
        fill=numpy_ops.infer_fill(prim_set) if fill is None else float(fill),
    )


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
        ValueError: If the column count does not match the tape, or if
            the tape holds an instruction the interpreter does not know.
    """
    if len(columns) != tape.columns:
        raise ValueError(f"The tape expects {tape.columns} columns, got {len(columns)}.")

    stack: list[Any] = []
    for step in range(tape.opcodes.size):
        opcode = int(tape.opcodes[step])
        operand = int(tape.operands[step])

        if opcode == Opcode.COL_LOAD:
            stack.append(columns[operand])
        elif opcode == Opcode.CONST:
            stack.append(float(tape.constants[operand]))
        elif opcode in _UNARY:
            stack[-1] = _UNARY[opcode](stack[-1])
        elif opcode in _UNARY_PROTECTED:
            stack[-1] = _UNARY_PROTECTED[opcode](stack[-1], fill=tape.fill)
        elif opcode in _WINDOWED:
            stack[-1] = _WINDOWED[opcode](stack[-1], operand)
        elif opcode in _BINARY:
            right = stack.pop()
            stack[-1] = _BINARY[opcode](stack[-1], right)
        elif opcode in _BINARY_PROTECTED:
            right = stack.pop()
            stack[-1] = _BINARY_PROTECTED[opcode](stack[-1], right, fill=tape.fill)
        elif opcode == Opcode.WHERE:
            on_false = stack.pop()
            on_true = stack.pop()
            stack[-1] = numpy_ops.vwhere(stack[-1], on_true, on_false)
        else:
            raise ValueError(
                f"Opcode {opcode} has no Python implementation. Consumer opcodes "
                f"are only available on the Numba backend."
            )

    return stack[-1]

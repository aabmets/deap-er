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

from collections.abc import Sequence
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy

if TYPE_CHECKING:
    from deap_er.private.typedefs import GPExprTypes

from .columnar import Window
from .numpy import numpy_ops
from .opcode_set import OPCODES_ARITY, USER_BASE, Opcode
from .primitives.primitive_nodes import Primitive
from .primitives.primitive_set_typed import PrimitiveSetTyped
from .primitives.primitive_tree import PrimitiveTree
from .tape import Tape, opcode_of

__all__: list[str] = [
    "child_indices",
    "immediate_windows",
    "postfix_order",
    "leaf_instruction",
    "lower_tree",
]


def child_indices(nodes: Sequence[Any]) -> list[list[int]]:
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


def immediate_windows(
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
        if not isinstance(node, Primitive) or opcode_of(node.name) >= USER_BASE:
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


def postfix_order(children: list[list[int]], folded: set[int]) -> list[int]:
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


def leaf_instruction(
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

    children = child_indices(nodes)
    windows, folded = immediate_windows(nodes, children)
    order = postfix_order(children, folded)
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
            opcode, operand = leaf_instruction(node, prim_set, pool, constants)
            pointer += 1
        else:
            opcode = opcode_of(node.name)
            operand = windows.get(index, -1)
            # Builtins are all in _ARITY, and they are the only nodes
            # that fold an operand into the instruction. A consumer
            # kernel takes every argument from the stack.
            pointer -= OPCODES_ARITY.get(opcode, node.arity) - 1
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

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
from collections.abc import Sequence
from math import gcd
from typing import Any, cast

from .columnar import Window
from .opcode_set import BUILTIN_OPCODES, USER_BASE
from .primitives.primitive_nodes import Primitive
from .primitives.primitive_set_typed import PrimitiveSetTyped
from .primitives.program_check import ProgramError, check_window
from .tape import opcode_of

__all__: list[str] = [
    "STEP_KEEPERS",
    "argument_strides",
    "window_scales",
    "scaled_window",
    "immediate_windows",
]

STEP_KEEPERS = frozenset({"delay", "diff"})
"""Window primitives whose output keeps the stride of their series."""


def argument_strides(prim_set: PrimitiveSetTyped) -> tuple[int, ...] | None:
    """Return the declared stride of every argument, or None when all are 1.

    Args:
        prim_set: Primitive set to read.

    Returns:
        One stride per argument, or None for a set without strides.
    """
    return cast(tuple[int, ...] | None, getattr(prim_set, "column_strides", None))


def window_scales(nodes: Sequence[Any], prim_set: PrimitiveSetTyped) -> dict[int, int]:
    """Return the row count of one window step for every builtin window node.

    A node's stride is how many rows each of its values is held for.
    A column has its declared stride, and a constant is neutral. A
    pointwise builtin takes the gcd of its children, because it changes
    whenever any of them does. ``delay`` and ``diff`` keep the stride of
    their series, and every other window primitive, consumer opcode and
    unknown primitive has stride 1. A window node's step is the gcd of
    its series arguments, or 1 when they are all constant.

    Args:
        nodes: Prefix-ordered tree nodes, with promoted primitives
            already expanded.
        prim_set: Primitive set the tree was built from.

    Returns:
        The step of each builtin window node by node index. Empty when
        the set declares no strides.
    """
    strides = argument_strides(prim_set)
    if strides is None:
        return {}
    scales: dict[int, int] = {}
    stack: list[int] = []
    for index in range(len(nodes) - 1, -1, -1):
        node = nodes[index]
        args = [stack.pop() for _ in range(node.arity)]
        if not isinstance(node, Primitive):
            stack.append(_leaf_stride(node, prim_set, strides))
            continue
        if node.name not in BUILTIN_OPCODES:
            stack.append(1)
            continue
        series = [arg for arg, kind in zip(args, node.args, strict=True) if kind is not Window]
        if len(series) == len(args):
            stack.append(gcd(*args))
            continue
        step = gcd(*series) or 1
        scales[index] = step
        stack.append(step if node.name in STEP_KEEPERS else 1)
    return scales


def _leaf_stride(node: Any, prim_set: PrimitiveSetTyped, strides: tuple[int, ...]) -> int:
    """Return a column's declared stride, or 0 for any other leaf."""
    value = getattr(node, "value", None)
    if isinstance(value, str) and value in prim_set.arguments:
        position = prim_set.arguments.index(value)
        return strides[position] if position < len(strides) else 1
    return 0


def scaled_window(name: str, length: int, step: int) -> int:
    """Convert a window of ``length`` steps to rows.

    Args:
        name: Name of the window primitive, for the error message.
        length: Window length in steps.
        step: Rows per step.

    Returns:
        The window length in rows.

    Raises:
        ProgramError: If the row count exceeds ``WINDOW_MAX``.
    """
    rows = length * step
    check_window(name, rows)
    return rows


def immediate_windows(
    nodes: Sequence[Any], children: list[list[int]], prim_set: PrimitiveSetTyped
) -> tuple[dict[int, int], set[int]]:
    """Fold the window argument of every builtin rolling node inline.

    A named window terminal is resolved through the set's evaluation
    context, as ``leaf_instruction`` does for numeric terminals. A
    window counts steps of its series, so it is scaled to rows by the
    step from ``window_scales``.

    Args:
        nodes: Tree nodes in prefix order.
        children: Child indices of every node.
        prim_set: Primitive set the tree was built from.

    Returns:
        The window length of each rolling node in rows, and the set of
        child indices that must not be pushed onto the stack.

    Raises:
        ProgramError: If a window argument is not a leaf, or if its
            value or its row count is not an integer in
            ``[1, WINDOW_MAX]``.
    """
    scales = window_scales(nodes, prim_set)
    windows: dict[int, int] = {}
    folded: set[int] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, Primitive) or opcode_of(node.name) >= USER_BASE:
            continue
        for position, arg_type in enumerate(node.args):
            if arg_type is not Window:
                continue
            child = children[index][position]
            length = _window_length(node, nodes[child], prim_set)
            windows[index] = scaled_window(node.name, length, scales.get(index, 1))
            folded.add(child)
    return windows, folded


def _window_length(node: Primitive, child: Any, prim_set: PrimitiveSetTyped) -> int:
    """Return the window length held by the window leaf ``child``."""
    if child.arity != 0:
        raise ProgramError(
            f"The window argument of '{node.name}' must be a leaf holding a "
            f"positive integer, so that it can be lowered to an immediate operand."
        )
    value = getattr(child, "value", None)
    if isinstance(value, str):
        value = prim_set.context.get(value)
    check_window(node.name, value)
    return int(cast(int, value))

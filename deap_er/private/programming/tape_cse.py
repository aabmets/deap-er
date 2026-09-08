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
from dataclasses import dataclass
from typing import Any

import numpy

from .opcode_set import OPCODES_ARITY, Opcode
from .opcodes import _apply_opcode, interpret_tape
from .tape import Tape

__all__: list[str] = ["build_cse_plan", "run_opcode_cse"]


@dataclass(frozen=True)
class CseNode:
    """One hash-consed postfix subexpression."""

    key: tuple[Any, ...]
    tape: Tape
    children: tuple[int, ...]


@dataclass(frozen=True)
class CsePlan:
    """Shared sub-tapes for one batch evaluation."""

    roots: tuple[int, ...]
    nodes: tuple[CseNode, ...]


def build_cse_plan(tapes: Sequence[Tape]) -> CsePlan:
    """Hash-cons postfix subexpressions across ``tapes``.

    Args:
        tapes: Tapes produced by ``lower_tree``.

    Returns:
        Root node ids per tape and the shared node table.
    """
    nodes: list[CseNode] = []
    key_to_id: dict[tuple[Any, ...], int] = {}
    roots: list[int] = []

    def intern(
        key: tuple[Any, ...],
        tape: Tape,
        children: tuple[int, ...],
    ) -> int:
        known = key_to_id.get(key)
        if known is not None:
            return known
        node_id = len(nodes)
        nodes.append(CseNode(key=key, tape=tape, children=children))
        key_to_id[key] = node_id
        return node_id

    for tape in tapes:
        stack: list[int] = []
        for step in range(tape.opcodes.size):
            opcode = int(tape.opcodes[step])
            operand = int(tape.operands[step])
            if opcode == int(Opcode.COL_LOAD):
                leaf = _leaf_tape(tape, step)
                key = (tape.fill, tape.columns, opcode, operand)
                stack.append(intern(key, leaf, ()))
                continue
            if opcode == int(Opcode.CONST):
                leaf = _leaf_tape(tape, step)
                value = float(tape.constants[operand])
                key = (tape.fill, tape.columns, opcode, value)
                stack.append(intern(key, leaf, ()))
                continue
            arity = OPCODES_ARITY[opcode]
            children = tuple(stack[-arity:])
            stack = stack[:-arity]
            child_keys = tuple(nodes[child].key for child in children)
            key = (tape.fill, tape.columns, opcode, operand, child_keys)
            child_tapes = tuple(nodes[child].tape for child in children)
            combined = _join_tapes(child_tapes, opcode, operand, tape)
            stack.append(intern(key, combined, children))
        roots.append(stack[-1])
    return CsePlan(roots=tuple(roots), nodes=tuple(nodes))


def run_opcode_cse(tapes: Sequence[Tape], matrix: numpy.ndarray) -> numpy.ndarray:
    """Evaluate ``tapes`` with shared sub-tape elimination.

    Args:
        tapes: Tapes to score.
        matrix: Packed ``(n_rows, n_columns)`` column table.

    Returns:
        ``(n_tapes, n_rows)`` results.
    """
    rows = matrix.shape[0]
    out = numpy.empty((len(tapes), rows), dtype=numpy.float64)
    if not tapes:
        return out
    plan = build_cse_plan(tapes)
    values = _evaluate_nodes_opcode(plan.nodes, matrix)
    for index, root in enumerate(plan.roots):
        out[index] = values[root]
    return out


def _leaf_tape(tape: Tape, step: int) -> Tape:
    """Slice one instruction into a standalone tape."""
    return Tape(
        opcodes=tape.opcodes[step : step + 1].copy(),
        operands=tape.operands[step : step + 1].copy(),
        constants=tape.constants.copy(),
        columns=tape.columns,
        depth=1,
        fill=tape.fill,
    )


def _join_tapes(
    parts: tuple[Tape, ...],
    opcode: int,
    operand: int,
    source: Tape,
) -> Tape:
    """Concatenate child tapes and append one parent instruction."""
    if not parts:
        raise ValueError("A combined tape needs at least one child tape.")
    constants: list[float] = []
    remap: dict[int, int] = {}
    opcodes: list[int] = []
    operands: list[int] = []
    for part in parts:
        for index, value in enumerate(part.constants):
            old = int(index)
            if old not in remap:
                remap[old] = len(constants)
                constants.append(float(value))
        opcodes.extend(int(code) for code in part.opcodes)
        for index, old_operand in enumerate(part.operands):
            if int(part.opcodes[index]) == int(Opcode.CONST):
                operands.append(remap[int(old_operand)])
            else:
                operands.append(int(old_operand))
    opcodes.append(opcode)
    operands.append(operand)
    return Tape(
        opcodes=numpy.array(opcodes, dtype=numpy.int32),
        operands=numpy.array(operands, dtype=numpy.int32),
        constants=numpy.array(constants, dtype=numpy.float64),
        columns=source.columns,
        depth=_tape_depth(opcodes),
        fill=source.fill,
    )


def _tape_depth(opcodes: list[int]) -> int:
    """Return the peak stack depth of a postfix tape."""
    pointer = 0
    depth = 0
    for opcode in opcodes:
        if opcode in {int(Opcode.COL_LOAD), int(Opcode.CONST)}:
            pointer += 1
        else:
            pointer -= OPCODES_ARITY[opcode] - 1
        depth = max(depth, pointer)
    return depth


def _evaluate_nodes_opcode(
    nodes: tuple[CseNode, ...],
    matrix: numpy.ndarray,
) -> list[numpy.ndarray]:
    """Evaluate unique nodes bottom-up with the Python oracle."""
    values: list[numpy.ndarray | None] = [None] * len(nodes)
    order = sorted(range(len(nodes)), key=lambda index: nodes[index].tape.opcodes.size)
    for index in order:
        node = nodes[index]
        if not node.children:
            values[index] = interpret_tape(node.tape, matrix)
            continue
        stack = [values[child] for child in node.children]
        opcode = int(node.tape.opcodes[-1])
        operand = int(node.tape.operands[-1])
        _apply_opcode(stack, matrix, node.tape, opcode, operand)
        values[index] = numpy.asarray(stack[-1], dtype=numpy.float64)
    for index, value in enumerate(values):
        if value is None:
            raise ValueError(f"CSE node {index} was not evaluated.")
    return values

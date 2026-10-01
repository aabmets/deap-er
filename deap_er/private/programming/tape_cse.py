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
from .opcodes import apply_opcode, interpret_tape
from .tape import Tape

__all__: list[str] = ["build_cse_plan", "evaluate_cse_nodes", "run_opcode_cse"]


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
    evaluate_cse_nodes(build_cse_plan(tapes), matrix, out)
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
    opcodes: list[int] = []
    operands: list[int] = []
    for part in parts:
        # Each part indexes its own pool, so shift its CONST operands
        # past the constants already copied from earlier parts.
        offset = len(constants)
        constants.extend(float(value) for value in part.constants)
        for code, old_operand in zip(part.opcodes, part.operands, strict=True):
            is_const = int(code) == int(Opcode.CONST)
            opcodes.append(int(code))
            operands.append(int(old_operand) + offset if is_const else int(old_operand))
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


def evaluate_cse_nodes(plan: CsePlan, matrix: numpy.ndarray, out: numpy.ndarray) -> None:
    """Evaluate unique nodes bottom-up with the Python oracle.

    Node ids are already topological, because ``build_cse_plan`` interns
    children before their parents. Each node's column is dropped after
    its last consumer has run, and each root is copied into ``out`` as
    soon as it exists. Peak memory therefore follows stack depth plus
    the nodes still shared with later tapes, not the node count.

    Args:
        plan: Plan built by ``build_cse_plan``.
        matrix: Packed ``(n_rows, n_columns)`` column table.
        out: ``(len(plan.roots), n_rows)`` array that receives one row
            per root.

    Raises:
        ValueError: If a node is consumed before it was evaluated.
    """
    pending = [0] * len(plan.nodes)
    for node in plan.nodes:
        for child in node.children:
            pending[child] += 1
    rows_of: dict[int, list[int]] = {}
    for index, root in enumerate(plan.roots):
        rows_of.setdefault(root, []).append(index)
    values: list[numpy.ndarray | None] = [None] * len(plan.nodes)
    for node_id, node in enumerate(plan.nodes):
        value = _evaluate_node(node_id, node, values, matrix)
        for index in rows_of.get(node_id, ()):
            out[index] = value
        for child in node.children:
            pending[child] -= 1
            if pending[child] == 0:
                values[child] = None
        if pending[node_id]:
            values[node_id] = value


def _evaluate_node(
    node_id: int,
    node: CseNode,
    values: list[numpy.ndarray | None],
    matrix: numpy.ndarray,
) -> numpy.ndarray:
    """Evaluate one node from its children's live values."""
    if not node.children:
        return numpy.asarray(interpret_tape(node.tape, matrix), dtype=numpy.float64)
    stack: list[Any] = []
    for child in node.children:
        if values[child] is None:
            raise ValueError(f"CSE node {child} was not evaluated before node {node_id}.")
        stack.append(values[child])
    opcode = int(node.tape.opcodes[-1])
    operand = int(node.tape.operands[-1])
    apply_opcode(stack, matrix, node.tape, opcode, operand)
    return numpy.asarray(stack[-1], dtype=numpy.float64)

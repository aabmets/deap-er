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

from typing import Any

from .compilers import compile_tree
from .opcode_set import USER_BASE
from .opcodes import lower_tree, numba_opcodes
from .primitives.primitive_nodes import Primitive, Terminal
from .primitives.primitive_set_typed import PrimitiveSetTyped
from .primitives.primitive_tree import PrimitiveTree

__all__: list[str] = [
    "body_is_columnar",
    "compile_body",
    "consume_node",
    "next_columnar_opcode",
    "rewrite_body",
    "used_arguments",
    "validate_subtree",
]

_MISSING_OPCODE = "no builtin opcode and no binding"


def validate_subtree(nodes: list[Any], prim_set: PrimitiveSetTyped) -> None:
    """Reject an incomplete, ill-typed, trivial, or ADF subtree.

    Args:
        nodes: Prefix-ordered subtree nodes.
        prim_set: Primitive set the subtree was built from.

    Raises:
        TypeError: If a node type does not match its parent slot.
        ValueError: If the slice is empty, incomplete, a lone
            argument terminal, or contains an ADF.
    """
    if not nodes:
        raise ValueError("An empty expression cannot be promoted.")
    node = nodes[0]
    value = getattr(node, "value", None)
    if len(nodes) == 1 and isinstance(value, str) and value in prim_set.arguments:
        raise ValueError("A lone argument terminal cannot be promoted.")
    if consume_node(nodes, 0, None, prim_set) != len(nodes):
        raise ValueError("The extracted nodes are not a complete typed tree.")


def consume_node(
    nodes: list[Any], index: int, expected: type | None, prim_set: PrimitiveSetTyped
) -> int:
    """Walk one prefix node and return the next unconsumed index.

    Args:
        nodes: Prefix-ordered subtree.
        index: Node to consume.
        expected: Required return type, or None at the root.
        prim_set: Primitive set used for closure and ADF checks.

    Returns:
        Index after this node and its descendants.

    Raises:
        TypeError: If the node type does not match ``expected``.
        ValueError: If the tree is incomplete, an ADF, or a
            primitive sits in a leaf-only type slot.
    """
    if index >= len(nodes):
        raise ValueError("The extracted nodes are not a complete typed tree.")
    node = nodes[index]
    if expected is not None:
        if not issubclass(node.ret, expected):
            raise TypeError(
                f"Primitive {node} return type {node.ret} "
                f"does not match the expected one: {expected}."
            )
        terminal_only = not prim_set.primitives[expected] and bool(prim_set.terminals[expected])
        if terminal_only and isinstance(node, Primitive):
            raise ValueError(f"Type '{expected}' is leaf-only; a primitive is not a valid closure.")
    if not isinstance(node, Primitive):
        return index + 1
    if node.name not in prim_set.context:
        raise ValueError("Cannot promote a subtree that contains an ADF.")
    pos = index + 1
    for arg in node.args:
        pos = consume_node(nodes, pos, arg, prim_set)
    return pos


def used_arguments(nodes: list[Any], prim_set: PrimitiveSetTyped) -> list[str]:
    """Collect argument terminals in primitive-set order.

    Args:
        nodes: Prefix-ordered subtree.
        prim_set: Primitive set that owns the argument names.

    Returns:
        Distinct argument names that appear in ``nodes``.
    """
    seen: set[str] = set()
    for node in nodes:
        value = getattr(node, "value", None)
        if isinstance(value, str) and value in prim_set.arguments:
            seen.add(value)
    return [name for name in prim_set.arguments if name in seen]


def rewrite_body(nodes: list[Any], used: list[str]) -> tuple[PrimitiveTree, list[str]]:
    """Rewrite used arguments to ``ARG0..ARGk-1``.

    Args:
        nodes: Prefix-ordered subtree on the parent set.
        used: Parent argument names, in set order.

    Returns:
        The rewritten body and its formal names.
    """
    rename = {old: f"ARG{index}" for index, old in enumerate(used)}
    formals = [rename[old] for old in used]
    body: list[Any] = []
    for node in nodes:
        value = getattr(node, "value", None)
        if isinstance(node, Terminal) and isinstance(value, str) and value in rename:
            body.append(Terminal(rename[value], True, node.ret))
        else:
            body.append(node)
    return PrimitiveTree(body), formals


def compile_body(
    body: PrimitiveTree,
    in_types: list[type],
    ret_type: type,
    prim_set: PrimitiveSetTyped,
) -> Any:
    """Compile the rewritten body against a dedicated set.

    Args:
        body: Body tree with ``ARG0..`` formals.
        in_types: Formal argument types.
        ret_type: Return type of the body.
        prim_set: Parent set whose context supplies callables.

    Returns:
        A callable implementing the body.
    """
    compiled = compile_tree(body, _body_set(in_types, ret_type, prim_set))
    if in_types:
        return compiled

    def constant() -> Any:
        return compiled

    return constant


def body_is_columnar(
    body: PrimitiveTree,
    in_types: list[type],
    ret_type: type,
    prim_set: PrimitiveSetTyped,
) -> bool:
    """Return whether ``body`` can be lowered to builtin or bound opcodes.

    Args:
        body: Rewritten body tree.
        in_types: Formal argument types.
        ret_type: Return type of the body.
        prim_set: Parent set whose context supplies callables.

    Returns:
        True when the body lowers. False when a primitive has no opcode.

    Raises:
        ValueError: If the body is malformed or otherwise illegal to
            lower, other than a missing opcode binding.
    """
    try:
        lower_tree(body, _body_set(in_types, ret_type, prim_set))
    except ValueError as err:
        if _MISSING_OPCODE in str(err):
            return False
        raise
    return True


def next_columnar_opcode(name: str) -> int:
    """Return a consumer opcode for ``name`` without binding it.

    Reuses a process-global binding of the same name so two sets that
    both allocate ``promo0`` do not collide.

    Args:
        name: Generated primitive name.

    Returns:
        The opcode to bind later.
    """
    known = numba_opcodes().get(name)
    if known is not None:
        return known
    bound = set(numba_opcodes().values())
    opcode = USER_BASE
    while opcode in bound:
        opcode += 1
    return opcode


def _body_set(
    in_types: list[type], ret_type: type, prim_set: PrimitiveSetTyped
) -> PrimitiveSetTyped:
    """Build a formal-only set that shares the parent evaluation context.

    Args:
        in_types: Formal argument types.
        ret_type: Return type of the body.
        prim_set: Parent set whose context supplies callables.

    Returns:
        A primitive set with one argument per formal.
    """
    body_set = PrimitiveSetTyped("_promo_body", in_types, ret_type)
    for key, value in prim_set.context.items():
        if key != "__builtins__" and key not in body_set.context:
            body_set.context[key] = value
    return body_set

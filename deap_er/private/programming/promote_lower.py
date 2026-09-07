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
from typing import Any

from .primitives.primitive_nodes import Primitive, Terminal
from .primitives.primitive_set_typed import PrimitiveSetTyped

__all__: list[str] = ["expand_promoted", "subtree_slice"]


def subtree_slice(nodes: Sequence[Any], begin: int) -> slice:
    """Return the slice of the subtree rooted at ``begin``.

    Args:
        nodes: Prefix-ordered tree nodes.
        begin: Index of the subtree root.

    Returns:
        Slice covering that subtree.
    """
    end = begin + 1
    total = nodes[begin].arity
    while total > 0:
        total += nodes[end].arity - 1
        end += 1
    return slice(begin, end)


def expand_promoted(nodes: Sequence[Any], prim_set: PrimitiveSetTyped) -> list[Any]:
    """Replace promoted primitives with their stored bodies.

    Children are expanded first. Each formal argument of a promoted
    body is substituted with a copy of the matching child subtree.
    Nested promoted names expand on a second pass of the substituted
    body. The input sequence is not mutated.

    Args:
        nodes: Prefix-ordered tree that may contain promoted names.
        prim_set: Primitive set that owns the promoted library.

    Returns:
        A new node list containing only non-promoted primitives and
        terminals, or a shallow copy of ``nodes`` when nothing is
        promoted.
    """
    library = getattr(prim_set, "promoted_library", None)
    if library is None or not library.records:
        return list(nodes)
    return _expand(list(nodes), library)


def _expand(nodes: list[Any], library: Any) -> list[Any]:
    """Expand one prefix tree against ``library``.

    Args:
        nodes: Prefix-ordered nodes of one subtree.
        library: Promoted-library object attached to the parent set.

    Returns:
        Expanded node list for that subtree.
    """
    if not nodes:
        return []
    root = nodes[0]
    children: list[list[Any]] = []
    pos = 1
    for _ in range(root.arity):
        bound = subtree_slice(nodes, pos)
        children.append(_expand(nodes[bound], library))
        pos = bound.stop
    record = None
    if isinstance(root, Primitive):
        record = library.records.get(root.name)
    if record is not None:
        return _expand(_substitute(record, children), library)
    out = [root]
    for child in children:
        out.extend(child)
    return out


def _substitute(record: Any, children: list[list[Any]]) -> list[Any]:
    """Fill a promoted body, replacing formals with child subtrees.

    Args:
        record: Promoted record with ``body`` and ``formals``.
        children: Expanded child subtrees, one per formal.

    Returns:
        Prefix-ordered nodes of the substituted body.
    """
    formals = {name: index for index, name in enumerate(record.formals)}
    out: list[Any] = []
    for node in record.body:
        if isinstance(node, Terminal) and isinstance(node.value, str) and node.value in formals:
            out.extend(children[formals[node.value]])
        else:
            out.append(node)
    return out

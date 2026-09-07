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
from typing import Any

from .compilers import clear_compile_cache
from .opcodes import bind_numba_opcode
from .primitives.primitive_set import PrimitiveSet
from .primitives.primitive_set_typed import PrimitiveSetTyped
from .primitives.primitive_tree import PrimitiveTree
from .promote_body import (
    body_is_columnar,
    compile_body,
    next_columnar_opcode,
    rewrite_body,
    used_arguments,
    validate_subtree,
)
from .promote_lower import expand_promoted
from .promote_store import (
    PromotedLibrary,
    PromotedRecord,
    detach_least_used,
    evict_least_used,
    library_of,
    next_promo_name,
    note_promoted_use,
    promoted_names,
    rollback_promote,
)

__all__: list[str] = [
    "PromotedLibrary",
    "PromotedRecord",
    "evict_least_used",
    "library_of",
    "next_promo_name",
    "note_promoted_use",
    "promote_subtree",
    "promoted_names",
]


def promote_subtree(
    prim_set: PrimitiveSetTyped,
    expr: PrimitiveTree | Sequence[Any],
    index: int = 0,
    *,
    max_library: int = 32,
    prefix: str = "promo",
    weight: float = 1.0,
) -> str:
    """Lift a complete typed subtree into ``prim_set`` as a primitive.

    The caller fires this helper. It does not run inside ``ea_*``.
    Formals are the set's argument terminals that appear in the
    subtree, in ``prim_set.arguments`` order. Constants and
    ephemerals stay baked into the body. Nested promoted names are
    expanded before the body is stored, so evicting an inner name
    does not leave ``USER_BASE`` in later tapes. On a columnar set
    the name is bound at or above ``USER_BASE`` and ``lower_tree``
    expands the body so tapes stay on builtin opcodes.

    Args:
        prim_set: Set that receives the new primitive.
        expr: Prefix tree holding the subtree.
        index: Root index of the subtree to lift.
        max_library: Maximum number of promoted names to keep.
        prefix: Prefix of the generated name (``promo0``, …).
        weight: Sampling weight of the new primitive.

    Returns:
        The generated primitive name.

    Raises:
        IndexError: If ``index`` is outside ``expr``.
        TypeError: If a node type does not match its parent slot.
        ValueError: If ``max_library`` is less than 1, if ``weight``
            is not greater than 0, if the slice is not a complete
            typed tree, if it is a lone argument terminal, if it
            contains an ADF, if the body cannot be lowered, or if
            an untyped set would receive a zero-arity primitive.
    """
    if max_library < 1:
        raise ValueError("max_library must be at least 1.")
    if weight <= 0:
        raise ValueError("Primitive weight must be greater than 0.")
    tree = expr if isinstance(expr, PrimitiveTree) else PrimitiveTree(expr)
    if index < 0 or index >= len(tree):
        raise IndexError(f"Subtree index {index} is outside the expression.")
    try:
        bound = tree.search_subtree(index)
    except IndexError as err:
        raise ValueError("The extracted nodes are not a complete typed tree.") from err
    if index == 0 and bound.stop != len(tree):
        raise ValueError("The extracted nodes are not a complete typed tree.")
    nodes = list(tree[bound])
    validate_subtree(nodes, prim_set)
    used = used_arguments(nodes, prim_set)
    if isinstance(prim_set, PrimitiveSet) and not used:
        raise ValueError("Untyped primitive sets require a primitive of arity at least 1.")
    in_types = [prim_set.mapping[name].ret for name in used]
    body, formals = rewrite_body(nodes, used)
    body = PrimitiveTree(expand_promoted(list(body), prim_set))
    func = compile_body(body, in_types, nodes[0].ret, prim_set)
    columnar = body_is_columnar(body, in_types, nodes[0].ret, prim_set)
    library = library_of(prim_set)
    detached: list[tuple[PromotedRecord, Any, Any]] = []
    name = ""
    try:
        while len(library.records) >= max_library:
            detached.append(detach_least_used(prim_set, library))
        name = next_promo_name(prim_set, library, prefix)
        if isinstance(prim_set, PrimitiveSet):
            prim_set.add_primitive(func, len(in_types), name=name, weight=weight)
        else:
            prim_set.add_primitive(func, in_types, nodes[0].ret, name=name, weight=weight)
        opcode = next_columnar_opcode(name) if columnar else None
        library.records[name] = PromotedRecord(
            name=name,
            uses=0,
            serial=library.serial,
            body=body,
            formals=formals,
            opcode=opcode,
        )
        library.generation += 1
        if opcode is not None:
            bind_numba_opcode(name, opcode)
        clear_compile_cache()
    except Exception:
        rollback_promote(prim_set, library, name, detached)
        raise
    return name

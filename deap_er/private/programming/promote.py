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
from dataclasses import dataclass, field
from typing import Any

from .compilers import clear_compile_cache
from .primitives.primitive_set import PrimitiveSet
from .primitives.primitive_set_typed import PrimitiveSetTyped
from .primitives.primitive_tree import PrimitiveTree
from .promote_body import (
    bind_if_columnar,
    compile_body,
    rewrite_body,
    used_arguments,
    validate_subtree,
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


@dataclass
class PromotedRecord:
    """One dynamically promoted primitive.

    Attributes:
        name: Generated primitive name.
        uses: How often generate or mutation sampled this name.
        serial: Allocation order, used to break eviction ties.
        body: Body tree whose formals are ``ARG0..ARGk-1``.
        formals: Formal argument names aligned with ``body``.
        opcode: Bound consumer opcode, or None when not columnar.
    """

    name: str
    uses: int
    serial: int
    body: PrimitiveTree
    formals: list[str]
    opcode: int | None


@dataclass
class PromotedLibrary:
    """Promoted primitives attached to a primitive set on first use."""

    serial: int = 0
    generation: int = 0
    records: dict[str, PromotedRecord] = field(default_factory=dict)


def promoted_names(prim_set: PrimitiveSetTyped) -> list[str]:
    """Return currently registered promoted names, oldest first.

    Args:
        prim_set: Primitive set that may hold a promoted library.

    Returns:
        Promoted names still on the set. Empty when none were added.
    """
    library = getattr(prim_set, "promoted_library", None)
    if library is None:
        return []
    return list(library.records)


def note_promoted_use(prim_set: PrimitiveSetTyped, name: str) -> None:
    """Increment the use count of a promoted name, if it is one.

    Args:
        prim_set: Primitive set that may hold a promoted library.
        name: Primitive name that was just sampled.
    """
    library = getattr(prim_set, "promoted_library", None)
    if library is None:
        return
    record = library.records.get(name)
    if record is not None:
        record.uses += 1


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
    ephemerals stay baked into the body. On a columnar set the name
    is bound at or above ``USER_BASE`` and ``lower_tree`` expands the
    body so tapes stay on builtin opcodes.

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
        ValueError: If ``max_library`` is less than 1, if the slice
            is not a complete typed tree, if it is a lone argument
            terminal, if it contains an ADF, or if an untyped set
            would receive a zero-arity primitive.
    """
    if max_library < 1:
        raise ValueError("max_library must be at least 1.")
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
    func = compile_body(body, in_types, nodes[0].ret, prim_set)
    library = library_of(prim_set)
    while len(library.records) >= max_library:
        evict_least_used(prim_set, library)
    name = next_promo_name(prim_set, library, prefix)
    opcode = bind_if_columnar(name, body, in_types, nodes[0].ret, prim_set)
    if isinstance(prim_set, PrimitiveSet):
        prim_set.add_primitive(func, len(in_types), name=name, weight=weight)
    else:
        prim_set.add_primitive(func, in_types, nodes[0].ret, name=name, weight=weight)
    library.records[name] = PromotedRecord(
        name=name,
        uses=0,
        serial=library.serial,
        body=body,
        formals=formals,
        opcode=opcode,
    )
    library.generation += 1
    clear_compile_cache()
    return name


def library_of(prim_set: PrimitiveSetTyped) -> PromotedLibrary:
    """Return the promoted library, attaching one on first use.

    Args:
        prim_set: Primitive set that will own the library.

    Returns:
        The set's ``promoted_library``.
    """
    library = getattr(prim_set, "promoted_library", None)
    if not isinstance(library, PromotedLibrary):
        library = PromotedLibrary()
        object.__setattr__(prim_set, "promoted_library", library)
    return library


def next_promo_name(prim_set: PrimitiveSetTyped, library: PromotedLibrary, prefix: str) -> str:
    """Allocate the next unused generated name.

    Args:
        prim_set: Set that must not already hold the name.
        library: Library whose serial is advanced.
        prefix: Name prefix.

    Returns:
        A name that is not an argument, context key, or mapping key.
    """
    while True:
        name = f"{prefix}{library.serial}"
        library.serial += 1
        if name in prim_set.arguments or name in prim_set.context or name in prim_set.mapping:
            continue
        return name


def evict_least_used(prim_set: PrimitiveSetTyped, library: PromotedLibrary) -> None:
    """Drop the least-used promoted name, oldest on a tie.

    Args:
        prim_set: Set that holds the primitive.
        library: Library that holds the record.
    """
    victim = min(library.records.values(), key=lambda rec: (rec.uses, rec.serial))
    prim = prim_set.mapping.pop(victim.name)
    for items in prim_set.primitives.values():
        while prim in items:
            items.remove(prim)
    prim_set.context.pop(victim.name, None)
    prim_set.prims_count -= 1
    del library.records[victim.name]

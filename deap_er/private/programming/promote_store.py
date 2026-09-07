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

from dataclasses import dataclass, field
from typing import Any

from .primitives.primitive_set import PrimitiveSet
from .primitives.primitive_set_typed import PrimitiveSetTyped
from .primitives.primitive_tree import PrimitiveTree

__all__: list[str] = [
    "PromotedLibrary",
    "PromotedRecord",
    "detach_least_used",
    "drop_from_lists",
    "evict_least_used",
    "library_of",
    "next_promo_name",
    "note_promoted_use",
    "promoted_names",
    "register_promoted",
    "rollback_promote",
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
    detach_least_used(prim_set, library)


def detach_least_used(
    prim_set: PrimitiveSetTyped, library: PromotedLibrary
) -> tuple[PromotedRecord, Any, Any]:
    """Remove the least-used promoted name and return enough to restore it.

    Args:
        prim_set: Set that holds the primitive.
        library: Library that holds the record.

    Returns:
        The record, the primitive node, and the Python callable.
    """
    victim = min(library.records.values(), key=lambda rec: (rec.uses, rec.serial))
    prim = prim_set.mapping.pop(victim.name)
    func = prim_set.context.pop(victim.name, None)
    drop_from_lists(prim_set, prim)
    prim_set.prims_count -= 1
    del library.records[victim.name]
    return victim, prim, func


def drop_from_lists(prim_set: PrimitiveSetTyped, prim: Any) -> None:
    """Remove ``prim`` from every primitive-type list.

    Args:
        prim_set: Set that holds the lists.
        prim: Primitive node to remove.
    """
    for items in prim_set.primitives.values():
        while prim in items:
            items.remove(prim)


def register_promoted(
    prim_set: PrimitiveSetTyped,
    func: Any,
    in_types: list[type],
    ret_type: type,
    name: str,
    weight: float,
) -> None:
    """Register a promoted callable on a typed or untyped set.

    Args:
        prim_set: Set that receives the primitive.
        func: Compiled body.
        in_types: Formal argument types.
        ret_type: Return type of the body.
        name: Generated primitive name.
        weight: Sampling weight.
    """
    if isinstance(prim_set, PrimitiveSet):
        PrimitiveSet.add_primitive(prim_set, func, len(in_types), name=name, weight=weight)
        return
    PrimitiveSetTyped.add_primitive(prim_set, func, in_types, ret_type, name=name, weight=weight)


def rollback_promote(
    prim_set: PrimitiveSetTyped,
    library: PromotedLibrary,
    name: str,
    detached: list[tuple[PromotedRecord, Any, Any]],
) -> None:
    """Undo a partial promote: drop the new name and restore evictions.

    Args:
        prim_set: Set that was being mutated.
        library: Promoted library on ``prim_set``.
        name: Generated name if one was allocated, else empty.
        detached: Evicted records to put back, oldest first.
    """
    if name and name in prim_set.mapping:
        prim = prim_set.mapping.pop(name)
        prim_set.context.pop(name, None)
        drop_from_lists(prim_set, prim)
        prim_set.prims_count -= 1
        library.records.pop(name, None)
    for record, prim, func in reversed(detached):
        register_promoted(prim_set, func, list(prim.args), prim.ret, record.name, prim.weight)
        library.records[record.name] = record

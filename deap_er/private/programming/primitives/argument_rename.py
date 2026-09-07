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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .primitive_set_typed import PrimitiveSetTyped

__all__: list[str] = ["apply_argument_renames"]


def _planned_renames(
    prim_set: PrimitiveSetTyped, mapping: dict[str, str]
) -> list[tuple[int, str, str]]:
    """Collect argument renames, ignoring missing keys and no-ops.

    Args:
        prim_set: Primitive set whose arguments may be renamed.
        mapping: Current argument names to new names.

    Returns:
        ``(index, old_name, new_name)`` triples in argument order.
    """
    planned: list[tuple[int, str, str]] = []
    for index, old_name in enumerate(prim_set.arguments):
        if old_name not in mapping:
            continue
        new_name = mapping[old_name]
        if new_name == old_name:
            continue
        planned.append((index, old_name, new_name))
    return planned


def _reject_rename_collisions(
    prim_set: PrimitiveSetTyped, planned: list[tuple[int, str, str]]
) -> None:
    """Reject a planned rename that would collide.

    Args:
        prim_set: Primitive set whose arguments would be renamed.
        planned: ``(index, old_name, new_name)`` triples to apply.

    Raises:
        ValueError: If two arguments take the same new name, or a new
            name is already an argument, primitive, or terminal that
            is not being vacated.
    """
    seen: set[str] = set()
    for new_name in (name for _, _, name in planned):
        if new_name in seen:
            raise ValueError(
                f"Argument name '{new_name}' is also an argument of the primitive set. "
                f"A compiled lambda cannot have duplicate parameter names."
            )
        seen.add(new_name)
    vacated = {old_name for _, old_name, _ in planned}
    for _, _, new_name in planned:
        if new_name in vacated:
            continue
        if new_name in prim_set.arguments:
            raise ValueError(
                f"Argument name '{new_name}' is also an argument of the primitive set. "
                f"A compiled lambda cannot have duplicate parameter names."
            )
        if new_name in prim_set.mapping or new_name in prim_set.context:
            raise ValueError(
                f"Argument name '{new_name}' is already used by a primitive or terminal. "
                f"A compiled lambda parameter would shadow the symbol."
            )


def apply_argument_renames(prim_set: PrimitiveSetTyped, mapping: dict[str, str]) -> None:
    """Rename input arguments on ``prim_set`` using ``mapping``.

    Names that are not current arguments are ignored. A no-op rename
    of an argument to itself is skipped. All collisions are checked
    before any argument is rewritten.

    Args:
        prim_set: Primitive set whose arguments are renamed in place.
        mapping: Current argument names to new names.

    Raises:
        ValueError: If a new name is already an argument, or is
            already used by a primitive or terminal.
    """
    planned = _planned_renames(prim_set, mapping)
    _reject_rename_collisions(prim_set, planned)
    updates: list[tuple[int, str, str, Any]] = [
        (index, old_name, new_name, prim_set.mapping[old_name])
        for index, old_name, new_name in planned
    ]
    for index, _, new_name, terminal in updates:
        prim_set.arguments[index] = new_name
        prim_set.mapping[new_name] = terminal
        terminal.value = new_name
    for _, old_name, _, terminal in updates:
        current = prim_set.mapping.get(old_name)
        if current is terminal:
            del prim_set.mapping[old_name]

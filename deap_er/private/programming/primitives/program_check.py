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
from numbers import Integral, Real
from typing import Any

from ..columnar import Window
from .primitive_nodes import Ephemeral, Primitive, Terminal
from .primitive_set_typed import PrimitiveSetTyped

__all__: list[str] = [
    "WINDOW_MAX",
    "ProgramError",
    "literal_matches",
    "check_window",
    "check_node",
    "slot_ephemeral",
    "ephemeral_with_value",
    "check_program",
]

WINDOW_MAX = 2**31 - 1
"""Largest window length that fits the ``int32`` operand of a tape."""


class ProgramError(ValueError, TypeError):
    """An expression is not a valid program for its primitive set.

    Raised for text that does not parse, for an empty expression, for a
    node whose type does not fit its slot, and for a window length that
    is not an integer in ``[1, WINDOW_MAX]``. It subclasses both
    ``ValueError`` and ``TypeError``, so callers that caught either of
    the errors raised before it existed keep working.
    """


def literal_matches(value: Any, ret_type: type) -> bool:
    """Return whether a literal value may occupy a ``ret_type`` slot.

    A ``Window`` slot takes a Python ``int``: ``Window`` is a type tag
    whose runtime value is an integer.

    Args:
        value: Literal value.
        ret_type: Type of the slot.

    Returns:
        True if the value fits the slot.
    """
    if ret_type is Window:
        return type(value) is int
    return issubclass(type(value), ret_type)


def check_window(owner: str | None, value: Any) -> None:
    """Raise unless ``value`` is a usable window length.

    Args:
        owner: Name of the primitive that takes the window, or None at
            the root.
        value: Window length.

    Raises:
        ProgramError: If ``value`` is not an integer in ``[1, WINDOW_MAX]``.
    """
    if not _is_integral(value) or not 1 <= value <= WINDOW_MAX:
        raise ProgramError(
            f"The window argument {value!r} of '{owner}' must be a leaf holding a "
            f"positive integer no larger than {WINDOW_MAX}."
        )


def _is_integral(value: Any) -> bool:
    """Return whether ``value`` is a real number with no fractional part.

    Integers are tested exactly: converting one too large for a float
    would raise ``OverflowError`` before the range check could refuse it.
    """
    if isinstance(value, Integral):
        return True
    if not isinstance(value, Real):
        return False
    try:
        return float(value).is_integer()
    except OverflowError:
        return False


def slot_ephemeral(prim_set: PrimitiveSetTyped, slot: type, value: Any) -> type[Ephemeral] | None:
    """Return the ephemeral class that a literal in ``slot`` restores as.

    Only numbers restore as ephemerals. When several ephemerals fit the
    slot, the first whose ``low`` / ``high`` sampler bounds hold ``value``
    wins, else the first one.

    Args:
        prim_set: Primitive set the slot belongs to.
        slot: Type of the slot.
        value: Literal value.

    Returns:
        The ephemeral class, or None if the slot has none.
    """
    if not isinstance(value, Real) or isinstance(value, bool):
        return None
    found = [
        item
        for item in prim_set.terminals.get(slot, ())
        if isinstance(item, type) and issubclass(item, Ephemeral)
    ]
    for item in found:
        low = getattr(item.func, "low", None)
        high = getattr(item.func, "high", None)
        if low is not None and high is not None and low <= value <= high:
            return item
    return found[0] if found else None


def ephemeral_with_value(class_: type[Ephemeral], value: Any) -> Ephemeral:
    """Return an instance of ``class_`` holding ``value`` without sampling.

    Args:
        class_: Ephemeral class registered on a primitive set.
        value: Value the instance holds.

    Returns:
        The ephemeral instance.
    """
    ret_type: Any = class_.ret
    node = class_.__new__(class_)
    Terminal.__init__(node, value, False, ret_type)
    return node


def check_node(node: Any, index: int) -> None:
    """Raise unless ``node`` is a ``Primitive`` or a ``Terminal``.

    Args:
        node: Tree element to check.
        index: Position of ``node`` in its tree, for the message.

    Raises:
        ProgramError: If ``node`` is any other object.
    """
    if not isinstance(node, (Primitive, Terminal)):
        raise ProgramError(
            f"Tree element {index} has type {type(node).__name__}, not Primitive or Terminal."
        )


def check_program(nodes: Sequence[Any], prim_set: PrimitiveSetTyped) -> None:
    """Check every leaf of a node sequence against its argument slot.

    Applies the rules of ``PrimitiveTree.from_string``, so that a tree
    which passes prints text that parses back. A leaf whose printed name
    is registered must return a subtype of its slot. Any other leaf must
    hold a value whose type fits the slot, or a number in a slot that
    has an ephemeral. A ``Window`` leaf must hold an integer in
    ``[1, WINDOW_MAX]``. The root, symbolic names that the set does not
    know, and malformed structure are left to the caller.

    Args:
        nodes: Tree nodes in prefix order.
        prim_set: Primitive set the tree was built from.

    Raises:
        ProgramError: If a node is not a ``Primitive`` or ``Terminal``,
            or if a leaf does not fit its slot.
    """
    mapping = prim_set.mapping
    slots: list[tuple[type, str]] = []
    for index, node in enumerate(nodes):
        if index and not slots:
            return
        check_node(node, index)
        slot, owner = slots.pop() if slots else (None, "")
        if isinstance(node, Primitive):
            name = node.name
            slots += [(arg, name) for arg in reversed(node.args)]
        elif slot is None:
            continue
        elif (
            slot is Window
            or node.conv_fct is not str
            or mapping.get(node.name) is not node
            or not issubclass(node.ret, slot)
        ):
            # A registered symbolic leaf in a matching slot needs no work.
            _check_leaf(node, slot, owner, prim_set)


def _check_leaf(node: Any, slot: type, owner: str, prim_set: PrimitiveSetTyped) -> None:
    """Raise if the leaf ``node`` does not fit the ``slot`` of ``owner``."""
    symbolic = node.conv_fct is str
    registered = prim_set.mapping.get(node.name if symbolic else node.format())
    if isinstance(registered, Terminal):
        if not issubclass(registered.ret, slot):
            raise ProgramError(
                f"Terminal {registered.name} return type {registered.ret} "
                f"does not match the expected one in '{owner}': {slot}."
            )
        value = registered.value
        if symbolic:
            value = prim_set.context.get(value)
    elif symbolic:
        return
    else:
        value = node.value
        fits = literal_matches(value, slot) or slot_ephemeral(prim_set, slot, value)
        if slot is not Window and not fits:
            raise ProgramError(
                f"Terminal {value!r} type {type(value)} does not match "
                f"the expected one in '{owner}': {slot}."
            )
    if slot is Window:
        check_window(owner, value)

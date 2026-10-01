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

import ast
import re
from typing import Any, cast

from ..columnar import Window
from .primitive_nodes import Primitive, Terminal
from .primitive_set_typed import PrimitiveSetTyped
from .program_check import (
    ProgramError,
    check_window,
    ephemeral_with_value,
    literal_matches,
    slot_ephemeral,
)

__all__: list[str] = ["Slot", "prefix_tokens", "primitive_from_token", "terminal_from_token"]

type Slot = tuple[type, str | None]

_TOKEN = re.compile(r"[(),]|[^\s(),]+")


def prefix_tokens(string: str, prim_set: PrimitiveSetTyped) -> list[str]:
    """Split a Python call expression into its node tokens in prefix order.

    Checks the parentheses and commas against the arity of each name
    registered on ``prim_set``. Unregistered tokens are literals and
    take no arguments. A zero-arity name may be written as ``name()``.

    Args:
        string: Python expression to split.
        prim_set: Primitive set that supplies the arities.

    Returns:
        The node tokens, without punctuation.

    Raises:
        ProgramError: If a primitive is not followed by ``(``, if its
            arguments are not separated by commas or closed by ``)``,
            if it gets too few or too many arguments, or if a token
            follows the complete expression.
    """
    tokens = _TOKEN.findall(string)
    names: list[str] = []
    owed: list[list[Any]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in "(),":
            raise ProgramError(f"Unexpected '{token}' where an argument was expected.")
        names.append(token)
        arity = getattr(prim_set.mapping.get(token), "arity", 0)
        index = _open_call(tokens, index + 1, token, arity, owed)
        if arity == 0:
            index = _close_calls(tokens, index, owed)
    if owed:
        raise ProgramError(f"Expression is incomplete; {owed[-1][0]} is missing arguments.")
    return names


def _peek(tokens: list[str], index: int) -> str | None:
    """Return the token at ``index``, or None past the end."""
    return tokens[index] if index < len(tokens) else None


def _open_call(tokens: list[str], index: int, name: str, arity: int, owed: list[list[Any]]) -> int:
    """Consume the opening of a call to ``name`` and return the next index."""
    following = _peek(tokens, index)
    if arity:
        if following != "(":
            raise ProgramError(f"Expression is incomplete; expected '(' after {name}.")
        owed.append([name, arity])
        return index + 1
    if following != "(":
        return index
    if _peek(tokens, index + 1) != ")":
        raise ProgramError(f"{name} takes no arguments.")
    return index + 2


def _close_calls(tokens: list[str], index: int, owed: list[list[Any]]) -> int:
    """Consume the separators after a complete argument and return the next index."""
    while owed:
        call = owed[-1]
        call[1] -= 1
        separator = _peek(tokens, index)
        if call[1]:
            if separator == ",":
                return index + 1
            if separator in (")", None):
                raise ProgramError(f"Expression is incomplete; {call[0]} is missing arguments.")
            raise ProgramError(f"Expected ',' between the arguments of {call[0]}, got {separator}.")
        if separator == ",":
            raise ProgramError(f"Unexpected extra argument to {call[0]}.")
        if separator != ")":
            raise ProgramError(f"Expression is incomplete; {call[0]} is missing ')'.")
        owed.pop()
        index += 1
    if index < len(tokens):
        raise ProgramError(f"Unexpected extra token after a complete expression: {tokens[index]}.")
    return index


def primitive_from_token(
    token: str, prim_set: PrimitiveSetTyped, slot: Slot
) -> Primitive | Terminal:
    """Resolve a token that is registered on ``prim_set``.

    Args:
        token: Primitive or terminal name.
        prim_set: Primitive set used to resolve names.
        slot: Expected return type and the name of the primitive that
            takes it. The name is None at the root.

    Returns:
        The registered primitive or terminal.

    Raises:
        ProgramError: If the return type does not match the slot, or if
            a ``Window`` slot gets an invalid window length.
    """
    ret_type, owner = slot
    primitive = cast(Primitive | Terminal, prim_set.mapping[token])
    if not issubclass(primitive.ret, ret_type):
        raise ProgramError(
            f"Primitive {primitive.name} return type {primitive.ret} does not match "
            f"the expected one {_where(owner)}: {ret_type}."
        )
    if ret_type is Window and isinstance(primitive, Terminal):
        value = primitive.value
        if primitive.conv_fct is str:
            value = prim_set.context.get(value)
        check_window(owner, value)
    return primitive


def terminal_from_token(token: str, prim_set: PrimitiveSetTyped, slot: Slot) -> Terminal:
    """Parse an unregistered token as a Python literal terminal.

    A ``Window`` slot accepts an ``int`` literal in ``[1, WINDOW_MAX]``.
    ``Window`` is a type tag whose runtime value is ``int``, and
    ``str(tree)`` writes those leaves as integers. A number restores as
    the set's ephemeral for the slot when there is one, so that
    ``mut_ephemeral`` can resample it. That is always the case for a
    window, and otherwise only for a number whose own type does not fit
    the slot.

    Args:
        token: Literal text from the expression.
        prim_set: Primitive set the expression is parsed against.
        slot: Expected type and the name of the primitive that takes it.

    Returns:
        A terminal or ephemeral wrapping the evaluated literal.

    Raises:
        ProgramError: If the token is not a Python literal, if its type
            does not match the slot, or if it is an invalid window.
    """
    ret_type, owner = slot
    try:
        value = ast.literal_eval(token)
    except (ValueError, SyntaxError) as err:
        raise ProgramError(f"Unable to evaluate terminal: {token}.") from err
    matches = literal_matches(value, ret_type)
    window = ret_type is Window
    if matches and not window:
        return Terminal(value, False, ret_type)
    ephemeral = None if window and not matches else slot_ephemeral(prim_set, ret_type, value)
    if not matches and ephemeral is None:
        raise ProgramError(
            f"Terminal {value} type {type(value)} does not match "
            f"the expected one {_where(owner)}: {ret_type}."
        )
    if window:
        check_window(owner, value)
    if ephemeral is None:
        return Terminal(value, False, ret_type)
    return ephemeral_with_value(ephemeral, value)


def _where(owner: str | None) -> str:
    """Describe the slot owned by ``owner`` for an error message."""
    return "at the root" if owner is None else f"in '{owner}'"

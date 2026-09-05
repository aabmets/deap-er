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
import copy
import re
from collections import deque
from collections.abc import Iterable
from typing import Any, cast, override

from .primitive_nodes import Primitive, Terminal
from .primitive_set_typed import PrimitiveSetTyped

__all__: list[str] = ["primitive_from_token", "terminal_from_token", "PrimitiveTree"]


def primitive_from_token(
    token: str, prim_set: PrimitiveSetTyped, ret_type: type | None
) -> Primitive | Terminal:
    """Resolve a token that is registered on ``prim_set``.

    Args:
        token: Primitive or terminal name.
        prim_set: Primitive set used to resolve names.
        ret_type: Expected return type, or None at the root.

    Returns:
        The registered primitive or terminal.

    Raises:
        TypeError: If the return type does not match ``ret_type``.
    """
    primitive = cast(Primitive | Terminal, prim_set.mapping[token])
    if ret_type is not None and not issubclass(primitive.ret, ret_type):
        raise TypeError(
            f"Primitive {primitive} return type {primitive.ret} "
            f"does not match the expected one: {ret_type}."
        )
    return primitive


def terminal_from_token(token: str, ret_type: type | None) -> Terminal:
    """Parse an unregistered token as a Python literal terminal.

    Args:
        token: Literal text from the expression.
        ret_type: Expected type, or None to take the literal's type.

    Returns:
        A terminal wrapping the evaluated literal.

    Raises:
        TypeError: If the token is not a Python literal, or if its
            type does not match ``ret_type``.
    """
    try:
        value = ast.literal_eval(token)
    except (ValueError, SyntaxError) as err:
        raise TypeError(f"Unable to evaluate terminal: {token}.") from err
    if ret_type is None:
        ret_type = type(value)
    if not issubclass(type(value), ret_type):
        raise TypeError(
            f"Terminal {value} type {type(value)} does not match the expected one: {ret_type}."
        )
    return Terminal(value, False, ret_type)


class PrimitiveTree(list[Any]):
    """Prefix-ordered tree of primitives and terminals.

    A list subclass used by genetic programming operators. Every node
    must expose an ``arity`` attribute.

    Args:
        content: Primitives and terminals that form the tree.
    """

    def __init__(self, content: Iterable[Any]) -> None:
        """Initialize the tree from ``content``."""
        super().__init__(content)

    def __deepcopy__(self, memo: dict[int, Any]) -> PrimitiveTree:
        """Return a deep copy of this tree.

        Args:
            memo: Memo mapping used by ``copy.deepcopy``.

        Returns:
            A new tree with copied contents and attributes.
        """
        new = self.__class__(self)
        new.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return new

    @override
    def __setitem__(self, key: Any, val: Any) -> None:  # type: ignore[override]
        """Replace a node or subtree, preserving tree arity.

        Args:
            key: Index or slice of the node or subtree to replace.
            val: Replacement node or sequence of nodes.

        Raises:
            IndexError: If a slice starts past the end of the tree.
            ValueError: If the replacement would change the tree arity.
        """
        if isinstance(key, slice):
            if key.start >= len(self):
                raise IndexError(
                    "Trying to set a slice larger than the size "
                    "of the PrimitiveTree is not allowed."
                )
            total = val[0].arity
            for node in val[1:]:
                total += node.arity - 1
            if total != 0:
                raise ValueError(
                    "Insertion of a subtree with an arity smaller "
                    "than the PrimitiveTree is not allowed."
                )
        elif val.arity != self[key].arity:
            raise ValueError(
                "PrimitiveTree node replacement with a node of a different arity is not allowed."
            )
        list.__setitem__(self, key, val)

    @override
    def __str__(self) -> str:
        """Return the tree as a Python expression string."""
        string = ""
        stack: list[Any] = []
        for node in self:
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()
                string = prim.format(*args)
                if len(stack) == 0:
                    break
                stack[-1][1].append(string)
        return str(string)

    @classmethod
    def from_string(cls, string: str, prim_set: PrimitiveSetTyped) -> PrimitiveTree:
        """Build a tree from a Python expression string.

        ``prim_set`` must contain every primitive that appears in
        ``string``.

        Args:
            string: Python expression to deserialize.
            prim_set: Primitive set used to resolve names.

        Returns:
            A tree populated with the deserialized primitives.

        Raises:
            TypeError: If a token is not a registered primitive
                and is not a Python literal, or if a primitive or
                terminal type does not match the expected type.
        """
        tokens = re.split("[ \t\n\r\f\v(),]", string)
        expr = []
        ret_types = deque()
        for token in tokens:
            if token == "":
                continue
            ret_type = ret_types.popleft() if ret_types else None
            if token in prim_set.mapping:
                primitive = primitive_from_token(token, prim_set, ret_type)
                expr.append(primitive)
                if isinstance(primitive, Primitive):
                    ret_types.extendleft(reversed(primitive.args))
                continue
            expr.append(terminal_from_token(token, ret_type))
        return cls(expr)

    def search_subtree(self, begin: int) -> slice:
        """Return the slice of the subtree rooted at ``begin``.

        Args:
            begin: Index of the subtree root.

        Returns:
            Slice covering that subtree.
        """
        end = begin + 1
        total = self[begin].arity
        while total > 0:
            total += self[end].arity - 1
            end += 1
        return slice(begin, end)

    @property
    def height(self) -> int:
        """Height of the tree, which is the depth of the deepest node."""
        stack = [0]
        max_depth = 0
        for elem in self:
            depth = stack.pop()
            max_depth = max(max_depth, depth)
            stack.extend([depth + 1] * elem.arity)
        return int(max_depth)

    @property
    def root(self) -> Any:
        """Root node of the tree (the first element)."""
        return self[0]

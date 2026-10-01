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

import copy
from collections import deque
from collections.abc import Iterable
from typing import Any, override

from .primitive_nodes import Primitive
from .primitive_set_typed import PrimitiveSetTyped
from .primitive_tokens import Slot, prefix_tokens, primitive_from_token, terminal_from_token
from .program_check import ProgramError

__all__: list[str] = ["PrimitiveTree"]


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
            start = 0 if key.start is None else key.start
            if start >= len(self):
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
        """Return the tree as a Python expression string.

        Raises:
            ProgramError: If a node follows the complete root
                expression, or if a primitive is missing arguments.
        """
        string = ""
        stack: list[Any] = []
        complete = False
        for node in self:
            if complete:
                raise ProgramError("The tree holds nodes after its complete root expression.")
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()
                string = prim.format(*args)
                if len(stack) == 0:
                    complete = True
                    break
                stack[-1][1].append(string)
        if stack:
            raise ProgramError("The tree is incomplete; a primitive is missing arguments.")
        return str(string)

    @classmethod
    def from_string(cls, string: str, prim_set: PrimitiveSetTyped) -> PrimitiveTree:
        """Build a tree from a Python expression string.

        ``prim_set`` must contain every primitive that appears in
        ``string``. The root must return a subtype of ``prim_set.ret``.
        A ``Window`` slot takes an ``int`` literal in ``[1, WINDOW_MAX]``.
        When the set has an ephemeral for a slot, a number in that slot
        restores as that ephemeral; see ``terminal_from_token``.

        Args:
            string: Python expression to deserialize.
            prim_set: Primitive set used to resolve names.

        Returns:
            A tree populated with the deserialized primitives.

        Raises:
            ProgramError: If the string is empty, if a token is not a
                registered primitive and is not a Python literal, if a
                primitive or terminal type does not match its slot (the
                root's slot is ``prim_set.ret``), if a window length is
                invalid, if a token arrives after the tree is complete,
                or if the stream still owes argument types.
                ``ProgramError`` subclasses ``ValueError`` and
                ``TypeError``.
        """
        tokens = prefix_tokens(string, prim_set)
        expr = []
        slots: deque[Slot] = deque([(prim_set.ret, None)])
        for token in tokens:
            if not slots:
                raise ProgramError(f"Unexpected extra token after a complete expression: {token}.")
            slot = slots.popleft()
            if token in prim_set.mapping:
                primitive = primitive_from_token(token, prim_set, slot)
                expr.append(primitive)
                if isinstance(primitive, Primitive):
                    slots.extendleft((arg, primitive.name) for arg in reversed(primitive.args))
                continue
            expr.append(terminal_from_token(token, prim_set, slot))
        if not expr:
            raise ProgramError("An empty string is not a program.")
        if slots:
            raise ProgramError("Expression is incomplete; missing arguments.")
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

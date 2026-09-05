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

__all__: list[str] = ["tree_to_infix"]

_INFIX = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "truediv": "/",
    "vadd": "+",
    "vsub": "-",
    "vmul": "*",
    "vdiv": "/",
}

_UNARY = {
    "neg": "-",
    "vneg": "-",
    "not": "not ",
    "vnot": "not ",
}


def _format_node(prim: Any, args: list[str]) -> str:
    """Format one node as infix text, or fall back to prefix.

    Args:
        prim: Primitive or terminal.
        args: Already formatted child expressions.

    Returns:
        Infix text for binary/unary arithmetic names, otherwise
        ``prim.format``.
    """
    if getattr(prim, "arity", 0) == 0:
        return str(prim.format())
    name = getattr(prim, "name", "")
    if name in _UNARY and len(args) == 1:
        return f"({_UNARY[name]}{args[0]})"
    if name in _INFIX and len(args) == 2:
        return f"({args[0]} {_INFIX[name]} {args[1]})"
    return str(prim.format(*args))


def tree_to_infix(expr: Any) -> str:
    """Return a tree as an infix expression string.

    Known arithmetic and logic names become infix operators. Other
    primitives stay in prefix ``name(args)`` form.

    Args:
        expr: Prefix-ordered tree of primitives and terminals.

    Returns:
        The expression in infix notation.
    """
    string = ""
    stack: list[tuple[Any, list[str]]] = []
    for node in expr:
        stack.append((node, []))
        while stack and len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            string = _format_node(prim, args)
            if not stack:
                break
            stack[-1][1].append(string)
    return str(string)

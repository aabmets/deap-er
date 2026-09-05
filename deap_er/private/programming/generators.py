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
from collections.abc import Callable
from inspect import isclass
from typing import Any

from deap_er.private.various.rng import rng

from .primitives.primitive_set_typed import PrimitiveSetTyped

__all__: list[str] = ["generate", "gen_full", "gen_grow", "gen_half_and_half"]

_ERR_MSG = "The gp.generate function tried to add a {0} of type '{1}', but there is none available."


def _choose_terminal(prim_set: PrimitiveSetTyped, ret_type: Any) -> Any:
    """Pick a terminal of ``ret_type`` from ``prim_set``.

    Args:
        prim_set: Primitive set to sample from.
        ret_type: Required return type of the terminal.

    Returns:
        A terminal instance, constructing class terminals on the fly.

    Raises:
        IndexError: If no terminal of ``ret_type`` is registered.
    """
    try:
        term = rng.choice(prim_set.terminals[ret_type])
    except IndexError as err:
        raise IndexError(_ERR_MSG.format("terminal", ret_type)) from err
    if isclass(term):
        return term()
    return term


def _choose_primitive(prim_set: PrimitiveSetTyped, ret_type: Any) -> Any:
    """Pick a primitive of ``ret_type`` from ``prim_set``.

    Args:
        prim_set: Primitive set to sample from.
        ret_type: Required return type of the primitive.

    Returns:
        A primitive from the set.

    Raises:
        IndexError: If no primitive of ``ret_type`` is registered.
    """
    try:
        return rng.choice(prim_set.primitives[ret_type])
    except IndexError as err:
        raise IndexError(_ERR_MSG.format("primitive", ret_type)) from err


def generate(
    prim_set: PrimitiveSetTyped,
    min_depth: int,
    max_depth: int,
    condition: Callable[..., bool],
    ret_type: Any | None = None,
) -> list[Any]:
    """Grow a tree as a depth-first list of primitives and terminals.

    Each branch grows until ``condition`` is true. A branch also stops
    early when its type has terminals but no primitives, which is how
    a strongly typed set expresses a leaf-only type such as a rolling
    window length. The list can be passed to ``PrimitiveTree`` to
    build a tree object.

    Args:
        prim_set: Primitive set from which nodes are selected.
        min_depth: Minimum depth of the random tree.
        max_depth: Maximum depth of the random tree.
        condition: Callable ``(height, depth)`` that decides when to
            stop growing a branch.
        ret_type: Return type of the generated tree. Defaults to
            ``prim_set.ret``.

    Returns:
        A tree as a flat list of primitives and terminals, in
        depth-first order.

    Raises:
        IndexError: If ``prim_set`` has no terminal or primitive of
            the required type.
    """
    if ret_type is None:
        ret_type = prim_set.ret
    expr = []
    height = rng.randint(min_depth, max_depth)
    stack = [(0, ret_type)]
    while len(stack) != 0:
        depth, ret_type = stack.pop()
        terminal_only = not prim_set.primitives[ret_type] and bool(prim_set.terminals[ret_type])
        if condition(height, depth) or terminal_only:
            expr.append(_choose_terminal(prim_set, ret_type))
        else:
            prim = _choose_primitive(prim_set, ret_type)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr


def gen_full(
    prim_set: PrimitiveSetTyped, min_depth: int, max_depth: int, ret_type: Any | None = None
) -> list[Any]:
    """Generate a full tree whose leaves share one depth.

    The common leaf depth is drawn between ``min_depth`` and
    ``max_depth``.

    Args:
        prim_set: Primitive set from which nodes are selected.
        min_depth: Minimum depth of the random tree.
        max_depth: Maximum depth of the random tree.
        ret_type: Return type of the generated tree. Defaults to
            ``prim_set.ret``.

    Returns:
        A full tree as a list of primitives and terminals.
    """

    def condition(height: int, depth: int) -> bool:
        return height == depth

    return generate(prim_set, min_depth, max_depth, condition, ret_type)


def gen_grow(
    prim_set: PrimitiveSetTyped, min_depth: int, max_depth: int, ret_type: Any | None = None
) -> list[Any]:
    """Generate a grown tree whose leaves may have different depths.

    Each leaf depth lies between ``min_depth`` and ``max_depth``.

    Args:
        prim_set: Primitive set from which nodes are selected.
        min_depth: Minimum depth of the random tree.
        max_depth: Maximum depth of the random tree.
        ret_type: Return type of the generated tree. Defaults to
            ``prim_set.ret``.

    Returns:
        A grown tree as a list of primitives and terminals.
    """

    def condition(height: int, depth: int) -> bool:
        cond = rng.random() < prim_set.terminal_ratio
        return depth == height or (depth >= min_depth and cond)

    return generate(prim_set, min_depth, max_depth, condition, ret_type)


def gen_half_and_half(
    prim_set: PrimitiveSetTyped, min_depth: int, max_depth: int, ret_type: Any | None = None
) -> list[Any]:
    """Generate a tree with either ``gen_grow`` or ``gen_full``.

    Args:
        prim_set: Primitive set from which nodes are selected.
        min_depth: Minimum depth of the random tree.
        max_depth: Maximum depth of the random tree.
        ret_type: Return type of the generated tree. Defaults to
            ``prim_set.ret``.

    Returns:
        Either a full tree or a grown tree, as a list of primitives
        and terminals.
    """
    choices = (gen_grow, gen_full)
    func = rng.choice(choices)
    return func(prim_set, min_depth, max_depth, ret_type)

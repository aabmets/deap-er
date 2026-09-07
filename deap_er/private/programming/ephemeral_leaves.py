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
from collections.abc import Sequence
from typing import Any

import numpy

from .columnar import Window
from .primitives.primitive_nodes import Ephemeral, Terminal
from .primitives.primitive_tree import PrimitiveTree
from .slim.slim_tree import SlimTree

__all__: list[str] = [
    "numeric_leaves",
    "extract_ephemerals",
    "assign_ephemerals",
    "leaf_range",
]

type LeafLoc = tuple[PrimitiveTree, int]


def numeric_leaves(individual: Any) -> list[LeafLoc]:
    """Return numeric-leaf locations in documented walk order.

    A numeric leaf is an ``Ephemeral`` or a ``Terminal`` whose
    ``ret is Window``. Order is prefix list order. A ``SlimTree``
    walks ``head``, then each delta, each in prefix order.

    Args:
        individual: ``PrimitiveTree`` or ``SlimTree`` to walk.

    Returns:
        ``(tree, index)`` pairs in contract order.
    """
    leaves: list[LeafLoc] = []
    if isinstance(individual, SlimTree):
        trees = [individual.head, *individual.deltas]
    else:
        trees = [individual]
    for tree in trees:
        for index, node in enumerate(tree):
            if isinstance(node, Ephemeral) or (isinstance(node, Terminal) and node.ret is Window):
                leaves.append((tree, index))
    return leaves


def extract_ephemerals(individual: Any) -> numpy.ndarray:
    """Copy numeric-leaf values into a vector in walk order.

    Args:
        individual: ``PrimitiveTree`` or ``SlimTree`` to read.

    Returns:
        A length-``n`` float array of current leaf values.
    """
    values = [float(tree[index].value) for tree, index in numeric_leaves(individual)]
    return numpy.asarray(values, dtype=float)


def assign_ephemerals(individual: Any, values: Sequence[float]) -> None:
    """Write repaired values onto numeric leaves, replacing each node.

    Window coordinates are rounded to ``int`` and clamped to the
    ephemeral's inclusive legal range, or to ``>= 1`` for a literal
    ``Window`` terminal. Nodes are replaced so a shallow clone does
    not alias values back onto the source tree.

    Args:
        individual: ``PrimitiveTree`` or ``SlimTree`` to write.
        values: Vector aligned with ``numeric_leaves``.

    Raises:
        ValueError: If ``values`` is the wrong length.
    """
    leaves = numeric_leaves(individual)
    if len(values) != len(leaves):
        raise ValueError(f"Expected {len(leaves)} numeric-leaf values, got {len(values)}.")
    for (tree, index), value in zip(leaves, values, strict=True):
        tree[index] = _replaced_leaf(tree[index], value)


def leaf_range(node: Any) -> tuple[float | None, float | None]:
    """Return the inclusive legal range of a numeric leaf.

    Window ephemerals use the ``low`` / ``high`` stamped on the
    sampler. A literal ``Window`` terminal is at least ``1``. Float
    ephemerals are unbounded.

    Args:
        node: Numeric leaf inspected by ``numeric_leaves``.

    Returns:
        ``(low, high)`` with ``None`` for an open end.
    """
    func = getattr(type(node), "func", None)
    low = getattr(func, "low", None)
    high = getattr(func, "high", None)
    if getattr(node, "ret", None) is Window:
        return (1.0 if low is None else float(low), None if high is None else float(high))
    return None, None


def _replaced_leaf(node: Any, value: float) -> Any:
    """Return an independent leaf carrying the repaired value."""
    replacement = copy.copy(node)
    if getattr(node, "ret", None) is Window:
        low, high = leaf_range(node)
        repaired: float | int = int(round(float(value)))
        if low is not None:
            repaired = max(int(low), repaired)
        if high is not None:
            repaired = min(int(high), repaired)
    else:
        repaired = float(value)
    replacement.value = repaired
    replacement.name = str(repaired)
    return replacement

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

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deap_er.private.typedefs import GPExprTypes, GPGraph
from deap_er.private.various.clone import clone_individual
from deap_er.private.various.rng import rng

from .primitives.primitive_nodes import Primitive

__all__: list[str] = ["build_tree_graph", "static_limit"]


def build_tree_graph(expr: GPExprTypes) -> GPGraph:
    """Build a graph representation of a tree expression.

    Args:
        expr: Tree expression to convert.

    Returns:
        Nodes, edges, and a mapping of node indices to labels.
    """
    nodes = list(range(len(expr)))
    edges = []
    stack = []
    labels = {}

    for i, node in enumerate(expr):
        if stack:
            edges.append((stack[-1][0], i))
            stack[-1][1] -= 1
        if isinstance(node, Primitive):
            labels[i] = node.name
        elif hasattr(node, "value"):
            labels[i] = node.value
        else:
            labels[i] = str(node)
        stack.append([i, getattr(node, "arity", 0)])
        while stack and stack[-1][1] == 0:
            stack.pop()

    return nodes, edges, labels


def static_limit(limiter: Callable[..., Any], max_value: int | float) -> Callable[..., Any]:
    """Return a decorator that rejects oversized GP offspring.

    May wrap crossover or mutation. An offspring whose measurement
    exceeds ``max_value`` is replaced by a randomly chosen parent.

    Args:
        limiter: Callable that measures an individual.
        max_value: Maximum allowed measurement.

    Returns:
        A decorator for a GP operator registered on a Toolbox.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> list[Any]:
            keep_inds = [clone_individual(ind) for ind in args]
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                if keep_inds and limiter(ind) > max_value:
                    new_inds[i] = clone_individual(rng.choice(keep_inds))
            return new_inds

        return wrapper

    return decorator

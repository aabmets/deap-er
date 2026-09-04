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
import random
from collections.abc import Callable
from copy import deepcopy
from functools import wraps
from typing import Any

from .dtypes import *
from .primitives import *

__all__ = ["compile_tree", "compile_adf_tree", "build_tree_graph", "static_limit"]


def compile_tree(expr: GPExprTypes, prim_set: PrimitiveSetTyped) -> Any:
    """Evaluate ``expr`` against ``prim_set``.

    Args:
        expr: Expression to compile. A string, a ``PrimitiveTree``,
            or any object whose string form is valid Python.
        prim_set: Primitive set that supplies the evaluation context.

    Returns:
        A callable if ``prim_set`` has one or more arguments,
        otherwise the result of the evaluation.

    Raises:
        MemoryError: If evaluation exceeds the recursion limit.
    """
    code = str(expr)
    if len(prim_set.arguments) > 0:
        args = ",".join(arg for arg in prim_set.arguments)
        code = f"lambda {args}: {code}"
    try:
        return eval(code, prim_set.context, {})
    except MemoryError as err:
        raise MemoryError(
            "Recursion depth of 90 exceeded. Use bloat control on your operators.\n"
        ) from err


def compile_adf_tree(expr: GPExprTypes, prim_sets: GPTypedSets) -> Any:
    """Compile a main tree together with its ADF trees.

    The first element of ``expr`` is the main tree. The rest are
    automatically defined functions that the main tree may call.

    Args:
        expr: Sequence of expressions to compile, one per primitive
            set. Each item may be a string, a ``PrimitiveTree``, or
            any object whose string form is valid Python.
        prim_sets: Primitive sets aligned with ``expr``. The first
            set is the main program and should refer to the ADFs;
            the following sets define those ADFs.

    Returns:
        A callable if the main primitive set has one or more
        arguments, otherwise the result of the evaluation.
    """
    adf_dict = dict()
    func = None
    for prim_set, sub_expr in reversed(list(zip(prim_sets, expr, strict=False))):
        prim_set.context.update(adf_dict)
        func = compile_tree(sub_expr, prim_set)
        adf_dict.update({prim_set.name: func})
    return func


def build_tree_graph(expr: GPExprTypes) -> GPGraph:
    """Build a graph representation of a tree expression.

    Args:
        expr: Tree expression to convert.

    Returns:
        Nodes, edges, and a mapping of node indices to labels.
    """
    nodes = list(range(len(expr)))
    edges = list()
    stack = list()
    labels = dict()

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
            keep_inds = [deepcopy(ind) for ind in args]
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                if keep_inds and limiter(ind) > max_value:
                    new_inds[i] = random.choice(keep_inds)
            return new_inds

        return wrapper

    return decorator

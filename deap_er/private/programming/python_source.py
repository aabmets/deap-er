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

import numpy

from .columnar import Window
from .primitives.primitive_set_typed import PrimitiveSetTyped

__all__: list[str] = ["SERIES_HELPER", "as_series", "python_source", "compile_python"]

SERIES_HELPER = "__deap_as_series"


def as_series(value: Any, reference: Any) -> Any:
    """Read a scalar window operand as a constant column.

    The tape backends treat a constant under a window primitive as a
    column of that constant. The Python backend calls this on every
    series operand of a window primitive so it agrees with them.

    Args:
        value: Series operand, or a scalar that stands for one.
        reference: A column whose length gives the row count.

    Returns:
        ``value`` unchanged when it is already a series or no row
        count is known, otherwise a ``float64`` column filled with it.
    """
    if numpy.ndim(value) != 0 or numpy.ndim(reference) == 0:
        return value
    return numpy.full(numpy.shape(reference)[0], value, dtype=numpy.float64)


def python_source(expr: Any, reference: str | None) -> tuple[str, bool]:
    """Render ``expr`` as Python source for the ``'python'`` backend.

    With a ``reference`` column name, each series operand of a window
    primitive (a primitive with a ``Window`` argument) is wrapped in
    :data:`SERIES_HELPER`, which :func:`as_series` must be bound to.

    Args:
        expr: Source text, or a prefix-ordered tree.
        reference: Argument name that supplies the row count, or
            ``None`` to render the plain expression.

    Returns:
        The source text, and whether it calls :data:`SERIES_HELPER`.
    """
    if isinstance(expr, str) or reference is None:
        return str(expr), False
    wrapped = False
    text = ""
    stack: list[tuple[Any, list[str]]] = []
    for node in expr:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            types = getattr(prim, "args", [])
            if Window in types:
                wrapped = True
                args = [
                    arg if type_ is Window else f"{SERIES_HELPER}({arg}, {reference})"
                    for arg, type_ in zip(args, types, strict=True)
                ]
            text = str(prim.format(*args))
            if not stack:
                break
            stack[-1][1].append(text)
    return text, wrapped


def compile_python(expr: Any, prim_set: PrimitiveSetTyped) -> Any:
    """Compile ``expr`` for the ``'python'`` backend.

    Args:
        expr: Source text, or a prefix-ordered tree.
        prim_set: Primitive set that supplies the evaluation context.

    Returns:
        A lambda over the set's arguments, or the evaluated result
        when the set has no arguments.

    Raises:
        MemoryError: If evaluation exceeds the recursion limit.
    """
    reference = prim_set.arguments[0] if prim_set.arguments else None
    code, wrapped = python_source(expr, reference)
    if reference is not None:
        code = f"lambda {','.join(prim_set.arguments)}: {code}"
    if wrapped:
        code = f"lambda {SERIES_HELPER}: {code}"
    compiled = _eval_python(code, prim_set)
    return compiled(as_series) if wrapped else compiled


def _eval_python(code: str, prim_set: PrimitiveSetTyped) -> Any:
    """Evaluate source text in the context of a primitive set.

    Args:
        code: Source text of the expression or of a lambda over it.
        prim_set: Primitive set that supplies the evaluation context.

    Returns:
        The evaluated object.

    Raises:
        MemoryError: If evaluation exceeds the recursion limit.
    """
    try:
        # nosemgrep: python.lang.security.audit.eval-detected.eval-detected
        return eval(code, prim_set.context, {})
    except MemoryError as err:
        raise MemoryError(
            "Recursion depth of 90 exceeded. Use bloat control on your operators.\n"
        ) from err

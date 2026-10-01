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

__all__: list[str] = ["SERIES_HELPER", "FLAT_DEPTH", "as_series", "python_source", "compile_python"]

SERIES_HELPER = "__deap_as_series"
FLAT_DEPTH = 64
_LOCAL_PREFIX = "__deap_t"
_PROGRAM = "__deap_program"
_BINDER = "__deap_bind"


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
    text, wrapped, _ = _render(expr, reference, None)
    return text, wrapped


def compile_python(expr: Any, prim_set: PrimitiveSetTyped) -> Any:
    """Compile ``expr`` for the ``'python'`` backend.

    A tree nested deeper than :data:`FLAT_DEPTH` compiles to a function
    that binds each primitive call to a local variable, because Python
    refuses source nested deeper than 200 parentheses.

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
    if not isinstance(expr, str) and _render(expr, reference, None)[2] > FLAT_DEPTH:
        statements: list[str] = []
        result, wrapped, _ = _render(expr, reference, statements)
        return _compile_flat(statements, result, wrapped, prim_set)
    code, wrapped = python_source(expr, reference)
    if reference is not None:
        code = f"lambda {','.join(prim_set.arguments)}: {code}"
    if wrapped:
        code = f"lambda {SERIES_HELPER}: {code}"
    compiled = _eval_python(code, prim_set)
    return compiled(as_series) if wrapped else compiled


def _render(
    expr: Any, reference: str | None, statements: list[str] | None
) -> tuple[str, bool, int]:
    """Render a prefix-ordered tree as Python source.

    Args:
        expr: Prefix-ordered tree.
        reference: Argument name that supplies the row count, or
            ``None`` to leave window operands unwrapped.
        statements: When given, each primitive call is appended to it
            as an assignment to a fresh local, and the local's name
            stands in for the call.

    Returns:
        The source text, whether it calls :data:`SERIES_HELPER`, and
        the depth of the tree.
    """
    wrapped = False
    text = ""
    depth = 0
    stack: list[tuple[Any, list[str]]] = []
    for node in expr:
        stack.append((node, []))
        depth = max(depth, len(stack))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            types = getattr(prim, "args", [])
            if reference is not None and Window in types:
                wrapped = True
                args = [
                    arg if type_ is Window else f"{SERIES_HELPER}({arg}, {reference})"
                    for arg, type_ in zip(args, types, strict=True)
                ]
            text = str(prim.format(*args))
            if statements is not None and prim.arity:
                local = f"{_LOCAL_PREFIX}{len(statements)}"
                statements.append(f"{local} = {text}")
                text = local
            if not stack:
                break
            stack[-1][1].append(text)
    return text, wrapped, depth


def _compile_flat(
    statements: list[str], result: str, wrapped: bool, prim_set: PrimitiveSetTyped
) -> Any:
    """Compile assignment statements into a function over the arguments.

    Args:
        statements: Assignments in evaluation order.
        result: Name or source text of the value to return.
        wrapped: Whether the statements call :data:`SERIES_HELPER`.
        prim_set: Primitive set that supplies the evaluation context.

    Returns:
        A function over the set's arguments, or its result when the
        set has no arguments.
    """
    lines = [f"def {_PROGRAM}({','.join(prim_set.arguments)}):"]
    lines += [f"    {statement}" for statement in statements]
    lines.append(f"    return {result}")
    if wrapped:
        lines = [f"def {_BINDER}({SERIES_HELPER}):", *(f"    {line}" for line in lines)]
        lines.append(f"    return {_PROGRAM}")
    namespace: dict[str, Any] = {}
    # nosemgrep: python.lang.security.audit.exec-detected.exec-detected
    exec("\n".join(lines), prim_set.context, namespace)
    program = namespace[_BINDER](as_series) if wrapped else namespace[_PROGRAM]
    return program if prim_set.arguments else program()


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

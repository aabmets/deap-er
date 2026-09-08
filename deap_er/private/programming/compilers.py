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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deap_er.private.typedefs import GPExprTypes, GPTypedSets

from deap_er.private.various.eval_cache import clear_eval_caches, invalidate_eval

from .compile_cache import CompileCache, compile_cache_key, expression_key
from .matrix_pack import as_matrix, is_packed_matrix
from .numba.numba_ops import bind_tape
from .opcodes import USER_BASE, interpret_tape, lower_tree
from .primitives.primitive_nodes import Primitive
from .primitives.primitive_set_typed import PrimitiveSetTyped
from .tree_graph import build_tree_graph, static_limit

__all__: list[str] = [
    "clear_compile_cache",
    "compile_tree",
    "compile_adf_tree",
    "build_tree_graph",
    "static_limit",
    "invalidate_compiled",
]

_COMPILE_CACHE_MAX = 1024
_compile_cache = CompileCache(_COMPILE_CACHE_MAX)


def clear_compile_cache() -> None:
    """Drop every compiled expression from the process-wide LRU cache.

    Call this after mutating a primitive set so a later
    ``compile_tree`` cannot return a lambda compiled against the
    previous context. Also clears every live ``EvalCache`` so a
    language mutation cannot keep stale fitness.
    """
    _compile_cache.clear()
    clear_eval_caches()


def _compile_python(code: str, prim_set: PrimitiveSetTyped) -> Any:
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


def _compile_tape(
    expr: GPExprTypes, prim_set: PrimitiveSetTyped, backend: str, dispatch: Any
) -> Any:
    """Lower an expression and bind it to a tape-running backend.

    Args:
        expr: Expression to lower.
        prim_set: Primitive set the expression was built from.
        backend: Either ``'opcode'`` or ``'numba'``.
        dispatch: Consumer kernel for the Numba backend, if any.

    Returns:
        A callable over the columns, or the result itself when the
        primitive set has no arguments.
    """
    tape = lower_tree(expr, prim_set)
    if backend == "opcode":
        consumer = tape.opcodes[tape.opcodes >= USER_BASE]
        if consumer.size:
            raise ValueError(
                f"Opcode {int(consumer[0])} belongs to a consumer kernel, which only "
                f"the numba backend can run. Use backend='python' or backend='numba'."
            )
    if backend == "numba":
        runner = bind_tape(tape, dispatch)
    else:

        def runner(*columns: Any) -> Any:
            if tape.columns == 0:
                return interpret_tape(tape, columns)
            if len(columns) == 1 and is_packed_matrix(columns[0]):
                matrix = as_matrix(columns, tape.columns)
                return interpret_tape(tape, matrix)
            return interpret_tape(tape, columns)

    if len(prim_set.arguments) == 0:
        return runner()
    return runner


def _reject_unknown_primitive(expr: GPExprTypes, prim_set: PrimitiveSetTyped) -> None:
    """Raise if a tree node names a primitive missing from ``context``.

    Args:
        expr: Expression about to be compiled.
        prim_set: Primitive set that must own every primitive name.

    Raises:
        NameError: If a primitive is not registered on the set.
    """
    if isinstance(expr, str):
        return
    for node in expr:
        if isinstance(node, Primitive) and node.name not in prim_set.context:
            raise NameError(f"The primitive '{node.name}' is not registered on the primitive set.")


def compile_tree(
    expr: GPExprTypes,
    prim_set: PrimitiveSetTyped,
    *,
    backend: str = "python",
    dispatch: Any = None,
) -> Any:
    """Evaluate ``expr`` against ``prim_set``.

    The default ``'python'`` backend evaluates the expression as source
    text. The ``'opcode'`` backend lowers the tree to a flat tape and
    runs it on a NumPy stack machine. The ``'numba'`` backend runs the
    same tape through a compiled interpreter and needs the ``numba``
    extra. Both tape backends require every primitive to carry an
    opcode, and reject the tree while lowering when one does not.

    Compiled results are cached, keyed by a structural tree key or
    the source text, the argument names, the contents of the primitive
    set, the backend, the dispatcher, and the promoted-library
    generation. ``clear_compile_cache`` drops the table;
    ``promote_subtree`` does that on every mutation.

    Args:
        expr: Expression to compile. A string, a ``PrimitiveTree``,
            or any object whose string form is valid Python.
        prim_set: Primitive set that supplies the evaluation context.
        backend: One of ``'python'``, ``'opcode'``, or ``'numba'``.
        dispatch: Compiled kernel that implements the consumer opcodes
            of the ``'numba'`` backend. Ignored by the other backends.

    Returns:
        A callable if ``prim_set`` has one or more arguments,
        otherwise the result of the evaluation.

    Raises:
        MemoryError: If evaluation exceeds the recursion limit.
        NameError: If ``expr`` holds a primitive missing from
            ``prim_set.context``.
        ValueError: If the backend is unknown, or if a tape backend
            cannot lower the expression.
    """
    library = getattr(prim_set, "promoted_library", None)
    generation = 0 if library is None else library.generation
    cache_key = compile_cache_key(
        backend, dispatch, expr, prim_set.arguments, prim_set.context, generation
    )
    cached = _compile_cache.get(cache_key)
    if cached is not None:
        return cached

    _reject_unknown_primitive(expr, prim_set)
    if backend == "python":
        code = str(expr)
        if len(prim_set.arguments) > 0:
            args = ",".join(prim_set.arguments)
            code = f"lambda {args}: {code}"
        compiled = _compile_python(code, prim_set)
    elif backend in ("opcode", "numba"):
        compiled = _compile_tape(expr, prim_set, backend, dispatch)
    else:
        raise ValueError(
            f"Unknown compile backend '{backend}'. Use 'python', 'opcode', or 'numba'."
        )

    _compile_cache.set(cache_key, compiled)
    return compiled


def invalidate_compiled(expr: Any) -> int:
    """Drop compile-cache and ``EvalCache`` entries for ``expr``.

    Matches a structural tree key, raw source text, or the historical
    ``lambda …: {expr}`` form. Pass a live tree, source text, or an
    ``expression_key`` captured before a mutation. ``tune_ephemerals``
    calls this after write-back so both the compile LRU and any live
    fitness cache drop the old expression.

    Args:
        expr: Expression whose cached compilations should be evicted.

    Returns:
        The number of compile-cache entries removed.
    """
    fragment = expr if isinstance(expr, str | tuple) else expression_key(expr)
    removed = _compile_cache.discard_expression(fragment)
    invalidate_eval(fragment)
    return removed


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
    adf_dict = {}
    func = None
    for prim_set, sub_expr in reversed(list(zip(prim_sets, expr, strict=False))):
        prim_set.context.update(adf_dict)
        func = compile_tree(sub_expr, prim_set)
        adf_dict.update({prim_set.name: func})
    return func

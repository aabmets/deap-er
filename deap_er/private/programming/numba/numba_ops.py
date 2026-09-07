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
from collections.abc import Callable, Sequence
from typing import Any

import numpy

from ..columnar import make_column_pset
from ..numpy.numpy_ops import add_numpy_primitives
from ..opcodes import USER_BASE, lower_tree
from ..primitives.primitive_tree import PrimitiveTree
from ..tape import Tape
from ..tape_batch import interpret_tapes
from .numba_compile import build

__all__: list[str] = [
    "USER_DISPATCH_SIGNATURE",
    "numba_available",
    "bind_tape",
    "reserve",
    "warmup_numba",
]

USER_DISPATCH_SIGNATURE = (
    "(op: int64, sp: int64, stack: float64[:, ::1], columns: float64[:, ::1], "
    "constants: float64[::1], scratch: float64[::1]) -> int64"
)
"""Signature every consumer dispatch kernel must have.

The interpreter keeps one stack of column-length rows. ``sp`` is the
number of occupied rows, so the top operand is ``stack[sp - 1]`` and a
kernel of arity ``n`` reads ``stack[sp - n]`` through ``stack[sp - 1]``,
writes its result into ``stack[sp - n]``, and returns ``sp - n + 1``::

    stack[sp - 1]  <- top operand      (n_rows values)
    stack[sp - 2]  <- second operand
    ...
    stack[sp - n]  <- first operand, and where the result goes
    stack[sp]      <- free scratch row

``constants`` is the tape constant pool, ``scratch`` is one spare row
of ``n_rows`` values that a kernel may clobber freely, and ``stack[sp]``
is always allocated and free to use as well. Builtin rolling primitives
fold their window length into the instruction, but a consumer primitive
receives every argument on the stack, so a window arrives as a
broadcast row and can be read from ``stack[sp - 1][0]``.
"""

_workspace: dict[str, numpy.ndarray] = {}


def numba_available() -> bool:
    """Report whether the optional Numba dependency is importable.

    Returns:
        True when the ``numba`` extra is installed.
    """
    try:
        import numba  # noqa: F401  # optional extra, probed without requiring it
    except ImportError:
        return False
    return True


def reserve(depth: int, rows: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Hand out the process-wide interpreter workspace.

    One workspace is shared by every compiled tape so that a population
    of cached programs cannot pin one large buffer each. It grows to
    fit the deepest tape seen so far and is replaced when the number of
    rows changes. Sharing it is what makes the compiled callables
    unsafe to run from several threads at once.

    Args:
        depth: Peak stack depth of the tape.
        rows: Number of samples per column.

    Returns:
        A stack of ``depth + 1`` rows, and one scratch row. The extra
        row is the one ``USER_DISPATCH_SIGNATURE`` promises consumer
        kernels at ``stack[sp]``.
    """
    stack = _workspace.get("stack")
    if stack is None or stack.shape[1] != rows or stack.shape[0] < depth + 1:
        stack = numpy.empty((depth + 1, rows), dtype=numpy.float64)
        _workspace["stack"] = stack
        _workspace["scratch"] = numpy.empty(rows, dtype=numpy.float64)
    return stack[: depth + 1], _workspace["scratch"]


def _as_matrix(columns: Sequence[Any], expected: int) -> numpy.ndarray:
    """Pack the arguments of a compiled tree into one contiguous matrix.

    Args:
        columns: Either one two-dimensional matrix, or one array per
            column.
        expected: Number of columns the tape expects.

    Returns:
        A C-contiguous ``(n_rows, n_columns)`` ``float64`` matrix.

    Raises:
        ValueError: If the column count does not match the tape.
    """
    if len(columns) == 1 and numpy.ndim(columns[0]) == 2:
        matrix = numpy.ascontiguousarray(columns[0], dtype=numpy.float64)
    else:
        if len(columns) != expected:
            raise ValueError(f"The tape expects {expected} columns, got {len(columns)}.")
        parts = [numpy.asarray(column, dtype=numpy.float64) for column in columns]
        matrix = numpy.ascontiguousarray(numpy.stack(parts, axis=1))
    if matrix.shape[1] != expected:
        raise ValueError(f"The tape expects {expected} columns, got {matrix.shape[1]}.")
    return matrix


def bind_tape(tape: Tape, dispatch: Any = None) -> Callable[..., numpy.ndarray]:
    """Bind a tape to the compiled interpreter.

    The interpreter is compiled once per process and once more for each
    distinct dispatcher, never per tree. It is single threaded on
    purpose: parallelize across individuals through ``toolbox.map``
    rather than inside a tree.

    Every compiled tape shares one process-wide workspace, so the
    returned callable must not be run from several threads at once. The
    result it returns is a fresh array.

    Args:
        tape: Tape produced by ``lower_tree``.
        dispatch: Compiled kernel implementing the opcodes at or above
            ``USER_BASE``, following ``USER_DISPATCH_SIGNATURE``.

    Returns:
        A callable that takes one array per column, or a single
        ``(n_rows, n_columns)`` matrix, and returns the result.

    Raises:
        ImportError: If the ``numba`` extra is not installed.
        ValueError: If the tape expects no columns, or if it holds
            consumer opcodes but no dispatcher was given.
    """
    if tape.columns == 0:
        raise ValueError(
            "The numba backend evaluates a tape over columns and cannot size a "
            "result without them. Use backend='python' or backend='opcode' for a "
            "primitive set that takes no arguments."
        )
    run, idle = build()
    if dispatch is None:
        unknown = tape.opcodes[tape.opcodes >= USER_BASE]
        if unknown.size:
            raise ValueError(
                f"The tape holds consumer opcode {int(unknown[0])} but no dispatch "
                "kernel was given. Pass dispatch= to compile_tree."
            )
        dispatch = idle

    def call(*columns: Any) -> numpy.ndarray:
        matrix = _as_matrix(columns, tape.columns)
        stack, scratch = reserve(tape.depth, matrix.shape[0])
        run(
            tape.opcodes,
            tape.operands,
            tape.constants,
            matrix,
            tape.fill,
            stack,
            scratch,
            dispatch,
        )
        return stack[0].copy()

    return call


def warmup_numba(*, parallel: bool = False, dispatch: Any = None) -> None:
    """Compile the Numba tape interpreter for this process.

    Runs a trivial column-load tape so the interpreter and the serial
    batch kernel are specialized before the first real evaluation.

    Args:
        parallel: If True, also specialize the ``prange`` batch kernel.
        dispatch: Consumer kernel to specialize. ``None`` uses the idle
            dispatcher.

    Raises:
        ImportError: If the ``numba`` extra is not installed.
    """
    pset = make_column_pset(["first"])
    add_numpy_primitives(pset)
    tape = lower_tree(PrimitiveTree([pset.mapping["first"]]), pset)
    matrix = numpy.zeros((2, 1), dtype=numpy.float64)
    interpret_tapes([tape], matrix, backend="numba", dispatch=dispatch)
    if parallel:
        interpret_tapes([tape], matrix, backend="numba", dispatch=dispatch, parallel=True)

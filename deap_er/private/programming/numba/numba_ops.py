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
from ..matrix_pack import DISPATCH_FLOAT32, MatrixPrecision, as_matrix, matrix_dtype
from ..numpy.numpy_ops import add_numpy_primitives
from ..opcodes import USER_BASE, lower_tree
from ..primitives.primitive_tree import PrimitiveTree
from ..tape import Tape, check_tape
from .numba_cache import ensure_numba_cache_dir
from .numba_compile import runner

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
broadcast row and can be read from ``stack[sp - 1][0]``. The columns
are always ``float64``, so a dispatch kernel cannot run with
``precision='float32'``.
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


def bind_tape(
    tape: Tape, dispatch: Any = None, *, precision: MatrixPrecision = "float64"
) -> Callable[..., numpy.ndarray]:
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
        precision: Storage dtype the callable packs its columns into,
            as in ``interpret_tapes``. A C-contiguous matrix of that
            dtype is used without a copy; the result is ``float64``.

    Returns:
        A callable that takes one array per column, or a single
        ``(n_rows, n_columns)`` matrix, and returns the result.

    Raises:
        ImportError: If the ``numba`` extra is not installed.
        ValueError: If the tape expects no columns, if it holds
            consumer opcodes but no dispatcher was given, if
            ``precision`` is unknown or is ``'float32'`` with a
            dispatcher, or if the tape is malformed (see
            ``check_tape``).
    """
    matrix_dtype(precision)
    if dispatch is not None and precision == "float32":
        raise ValueError(DISPATCH_FLOAT32)
    if tape.columns == 0:
        raise ValueError(
            "The numba backend evaluates a tape over columns and cannot size a "
            "result without them. Use backend='python' or backend='opcode' for a "
            "primitive set that takes no arguments."
        )
    if dispatch is None:
        unknown = tape.opcodes[tape.opcodes >= USER_BASE]
        if unknown.size:
            raise ValueError(
                f"The tape holds consumer opcode {int(unknown[0])} but no dispatch "
                "kernel was given. Pass dispatch= to compile_tree."
            )
    run = runner(dispatch)
    check_tape(tape)

    def call(*columns: Any) -> numpy.ndarray:
        matrix = as_matrix(columns, tape.columns, precision)
        stack, scratch = reserve(tape.depth, matrix.shape[0])
        run(
            tape.opcodes,
            tape.operands,
            tape.constants,
            matrix,
            tape.fill,
            stack,
            scratch,
        )
        return stack[0].copy()

    return call


def warmup_numba(
    *,
    parallel: bool = False,
    dispatch: Any = None,
    precisions: Sequence[MatrixPrecision] = ("float64",),
) -> None:
    """Compile the Numba tape interpreter for this process.

    Runs a trivial column-load tape so the interpreter and the serial
    batch kernel are specialized before the first real evaluation.
    When ``NUMBA_CACHE_DIR`` is unset, a writable directory under the
    user's cache (``~/.cache/deap-er/numba``, or
    ``$XDG_CACHE_HOME/deap-er/numba``) is used so spawned workers with
    a different working directory can reload the interpreter from disk
    instead of recompiling it.

    Args:
        parallel: If True, also specialize the ``prange`` batch kernel.
        dispatch: Consumer kernel to specialize. ``None`` specializes
            the builtin interpreter only.
        precisions: Matrix storage dtypes to specialize for, any of
            ``'float64'`` and ``'float32'``. A ``float32`` matrix needs
            its own compiled kernels, which also go to the disk cache.

    Raises:
        ImportError: If the ``numba`` extra is not installed.
        ValueError: If a precision is unknown, or if ``'float32'`` is
            asked for together with ``dispatch``.
    """
    for precision in precisions:
        matrix_dtype(precision)
    if dispatch is not None and "float32" in precisions:
        raise ValueError(DISPATCH_FLOAT32)
    ensure_numba_cache_dir()
    pset = make_column_pset(["first"])
    add_numpy_primitives(pset)
    tape = lower_tree(PrimitiveTree([pset.mapping["first"]]), pset)
    for precision in precisions:
        matrix = numpy.zeros((2, 1), dtype=matrix_dtype(precision))
        # bind_tape raises the ImportError when the numba extra is missing.
        bind_tape(tape, dispatch, precision=precision)(matrix)
        # Optional extra: numba_batch imports numba at module level. The
        # serial batch path of interpret_tapes runs the CSE plan and would
        # leave these kernels uncompiled, so they are launched directly.
        from .numba_batch import launch_kernels

        launch_kernels([tape], matrix, dispatch, False)
        if parallel:
            launch_kernels([tape], matrix, dispatch, True)

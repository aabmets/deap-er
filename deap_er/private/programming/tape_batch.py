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
from collections.abc import Iterable
from typing import Any

import numpy

from .opcodes import USER_BASE, interpret_tape
from .tape import Tape

__all__: list[str] = ["interpret_tapes"]

_MATRIX_SHAPE = (
    "interpret_tapes expects a packed (n_rows, n_columns) matrix, not a sequence of columns."
)
_NUMBA_MISSING = (
    "The numba backend needs the optional 'numba' dependency. "
    "Install it with: pip install deap-er[numba]"
)


def _as_batch_matrix(matrix: Any) -> numpy.ndarray:
    """Validate and pack the batch input matrix.

    Args:
        matrix: Caller-supplied column table.

    Returns:
        A C-contiguous ``(n_rows, n_columns)`` ``float64`` matrix.

    Raises:
        ValueError: If ``matrix`` is a sequence of columns or is not
            two-dimensional.
    """
    if isinstance(matrix, (list, tuple)):
        raise ValueError(_MATRIX_SHAPE)
    packed = numpy.asarray(matrix)
    if packed.ndim != 2:
        raise ValueError(
            f"interpret_tapes expects a packed (n_rows, n_columns) matrix, got ndim={packed.ndim}."
        )
    return numpy.ascontiguousarray(packed, dtype=numpy.float64)


def _check_tapes(tapes: tuple[Tape, ...], columns: int) -> None:
    """Reject tapes that do not match the packed matrix.

    Args:
        tapes: Tapes about to run.
        columns: Column count of the packed matrix.

    Raises:
        ValueError: If a tape expects a different column count.
    """
    for tape in tapes:
        if tape.columns != columns:
            raise ValueError(f"The tape expects {tape.columns} columns, got {columns}.")


def _reject_consumer(tapes: tuple[Tape, ...]) -> None:
    """Reject tapes that hold a consumer opcode.

    Args:
        tapes: Tapes about to run on the opcode backend.

    Raises:
        ValueError: If any tape holds an opcode at or above
            ``USER_BASE``.
    """
    for tape in tapes:
        consumer = tape.opcodes[tape.opcodes >= USER_BASE]
        if consumer.size:
            raise ValueError(
                f"Opcode {int(consumer[0])} belongs to a consumer kernel, which only "
                f"the numba backend can run. Use backend='numba'."
            )


def _write_opcode_row(out: numpy.ndarray, index: int, result: Any, rows: int) -> None:
    """Broadcast one opcode-backend result into an output row.

    Args:
        out: Batch result of shape ``(n_tapes, n_rows)``.
        index: Row to write.
        result: Scalar or column returned by ``interpret_tape``.
        rows: Expected row length.
    """
    values = numpy.asarray(result, dtype=numpy.float64)
    out[index] = numpy.broadcast_to(values, (rows,))


def _run_opcode(tapes: tuple[Tape, ...], matrix: numpy.ndarray) -> numpy.ndarray:
    """Run every tape on the NumPy stack machine.

    Args:
        tapes: Tapes to evaluate.
        matrix: Packed column table.

    Returns:
        ``(n_tapes, n_rows)`` results.
    """
    _reject_consumer(tapes)
    rows = matrix.shape[0]
    parts = [matrix[:, column] for column in range(matrix.shape[1])]
    out = numpy.empty((len(tapes), rows), dtype=numpy.float64)
    for index, tape in enumerate(tapes):
        _write_opcode_row(out, index, interpret_tape(tape, parts), rows)
    return out


def interpret_tapes(
    tapes: Iterable[Tape],
    matrix: Any,
    *,
    backend: str = "opcode",
    dispatch: Any = None,
    parallel: bool = False,
) -> numpy.ndarray:
    """Run many tapes over one packed column matrix.

    The result has one row per tape and one column per sample. It is a
    new array: it does not alias ``matrix`` or the interpreter
    workspace. Cache unique programs by ``str(tree)`` and lower the
    tree object — ``PrimitiveTree.from_string`` cannot round-trip a
    ``Window`` ephemeral.

    The ``'opcode'`` backend unpacks the matrix columns once and runs
    the NumPy stack machine. The ``'numba'`` backend is a compiled
    loop over the same tapes. ``parallel=True`` evaluates tapes on
    several threads, each with its own workspace of shape
    ``(depth + 1, n_rows)``. It is ignored when Numba reports one
    thread. ``parallel=True`` is not available on the opcode backend.

    Args:
        tapes: Tapes produced by ``lower_tree``, in the order of the
            output rows. A one-shot iterable is consumed once.
        matrix: Packed ``(n_rows, n_columns)`` column table. A sequence
            of columns is rejected.
        backend: Either ``'opcode'`` or ``'numba'``.
        dispatch: Compiled kernel that implements the consumer opcodes
            of the ``'numba'`` backend. Ignored by ``'opcode'``.
            With ``parallel=True`` it must be safe on several stacks
            at once — no process-global buffer.
        parallel: If True, run the Numba path with one workspace per
            thread. Requires ``backend='numba'``.

    Returns:
        A C-contiguous ``float64`` array of shape
        ``(len(tapes), n_rows)``.

    Raises:
        ValueError: If the backend is unknown, if ``parallel`` is set
            on the opcode backend, if ``matrix`` is not a packed
            table, or if a tape does not match the matrix.
        ImportError: If ``backend='numba'`` and the ``numba`` extra
            is not installed.
    """
    tapes = tuple(tapes)
    packed = _as_batch_matrix(matrix)
    _check_tapes(tapes, packed.shape[1])
    if backend == "opcode":
        if parallel:
            raise ValueError("parallel=True requires backend='numba'.")
        return _run_opcode(tapes, packed)
    if backend == "numba":
        try:
            # Optional extra: imported only when the Numba batch path is asked for.
            from .numba.numba_batch import run_tapes
        except ImportError as err:
            raise ImportError(_NUMBA_MISSING) from err
        return run_tapes(tapes, packed, dispatch=dispatch, parallel=parallel)
    raise ValueError(f"Unknown compile backend '{backend}'. Use 'opcode' or 'numba'.")

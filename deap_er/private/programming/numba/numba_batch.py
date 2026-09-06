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
from collections.abc import Sequence
from typing import Any

import numba  # optional extra; this module is imported only for backend='numba'
import numpy

from ..opcodes import USER_BASE
from ..tape import Tape
from . import numba_kernels
from .numba_compile import build
from .numba_ops import reserve

__all__: list[str] = ["run_tapes"]

_NO_COLUMNS = (
    "The numba backend evaluates a tape over columns and cannot size a "
    "result without them. Use backend='python' or backend='opcode' for a "
    "primitive set that takes no arguments."
)

_batch: dict[str, Any] = {}


def interpret_many_parallel(  # pragma: no cover
    run: Any,
    opcodes: Any,
    operands: Any,
    constants: Any,
    op_starts: Any,
    op_lens: Any,
    c_starts: Any,
    c_lens: Any,
    fills: Any,
    columns: Any,
    stacks: Any,
    scratches: Any,
    dispatch: Any,
    out: Any,
) -> None:
    """Run many tapes with one workspace per thread.

    Args:
        run: Compiled single-tape interpreter.
        opcodes: Concatenated instruction stream.
        operands: Concatenated immediates.
        constants: Concatenated constant pools.
        op_starts: Start index of each tape in ``opcodes``.
        op_lens: Instruction count of each tape.
        c_starts: Start index of each tape in ``constants``.
        c_lens: Constant-pool length of each tape.
        fills: Protected-op fill of each tape.
        columns: Packed input matrix.
        stacks: Per-thread workspaces.
        scratches: Per-thread scratch rows.
        dispatch: Consumer kernel.
        out: Result of shape ``(n_tapes, n_rows)``.
    """
    rows = columns.shape[0]
    for index in numba.prange(op_starts.shape[0]):  # ty: ignore[not-iterable]
        slot = numba.get_thread_id()
        start = op_starts[index]
        n_ops = op_lens[index]
        const_start = c_starts[index]
        n_consts = c_lens[index]
        run(
            opcodes[start : start + n_ops],
            operands[start : start + n_ops],
            constants[const_start : const_start + n_consts],
            columns,
            fills[index],
            stacks[slot],
            scratches[slot],
            dispatch,
        )
        for row in range(rows):
            out[index, row] = stacks[slot, 0, row]


def _batch_kernels() -> tuple[Any, Any, Any, Any]:
    """Compile the batch interpreters once per process.

    Returns:
        Single-tape runner, idle dispatcher, serial batch kernel,
        and parallel batch kernel.
    """
    run, idle = build()
    if "many" not in _batch:
        jit = numba.njit(cache=False, nogil=True, error_model="numpy")
        _batch["many"] = jit(numba_kernels.interpret_many)
        _batch["many_parallel"] = numba.njit(
            cache=False, nogil=True, error_model="numpy", parallel=True
        )(interpret_many_parallel)
    return run, idle, _batch["many"], _batch["many_parallel"]


def _pack(tapes: Sequence[Tape]) -> dict[str, numpy.ndarray | int]:
    """Concatenate tapes into jagged streams plus offsets.

    Args:
        tapes: Tapes to pack. Must not be empty.

    Returns:
        Arrays consumed by the compiled batch kernels.
    """
    op_lens = numpy.array([tape.opcodes.size for tape in tapes], dtype=numpy.int64)
    c_lens = numpy.array([tape.constants.size for tape in tapes], dtype=numpy.int64)
    op_starts = numpy.zeros(len(tapes), dtype=numpy.int64)
    c_starts = numpy.zeros(len(tapes), dtype=numpy.int64)
    op_starts[1:] = numpy.cumsum(op_lens[:-1])
    c_starts[1:] = numpy.cumsum(c_lens[:-1])
    if int(op_lens.sum()) == 0:
        opcodes = numpy.empty(0, dtype=numpy.int32)
        operands = numpy.empty(0, dtype=numpy.int32)
    else:
        opcodes = numpy.concatenate([tape.opcodes for tape in tapes]).astype(
            numpy.int32, copy=False
        )
        operands = numpy.concatenate([tape.operands for tape in tapes]).astype(
            numpy.int32, copy=False
        )
    if int(c_lens.sum()) == 0:
        constants = numpy.empty(0, dtype=numpy.float64)
    else:
        constants = numpy.concatenate([tape.constants for tape in tapes]).astype(
            numpy.float64, copy=False
        )
    return {
        "opcodes": opcodes,
        "operands": operands,
        "constants": constants,
        "op_starts": op_starts,
        "op_lens": op_lens,
        "c_starts": c_starts,
        "c_lens": c_lens,
        "fills": numpy.array([tape.fill for tape in tapes], dtype=numpy.float64),
        "max_depth": max(tape.depth for tape in tapes),
    }


def run_tapes(
    tapes: Sequence[Tape],
    matrix: numpy.ndarray,
    dispatch: Any = None,
    parallel: bool = False,
) -> numpy.ndarray:
    """Evaluate tapes on the compiled interpreter.

    Args:
        tapes: Tapes produced by ``lower_tree``.
        matrix: C-contiguous ``(n_rows, n_columns)`` ``float64`` table.
        dispatch: Consumer kernel, or None to use the idle dispatcher.
        parallel: If True, use one workspace per Numba thread when
            more than one thread is available.

    Returns:
        ``(n_tapes, n_rows)`` results.

    Raises:
        ValueError: If a tape has no columns, or holds a consumer
            opcode without a dispatcher.
    """
    rows = matrix.shape[0]
    out = numpy.empty((len(tapes), rows), dtype=numpy.float64)
    if not tapes:
        return out
    for tape in tapes:
        if tape.columns == 0:
            raise ValueError(_NO_COLUMNS)
        unknown = tape.opcodes[tape.opcodes >= USER_BASE]
        if unknown.size and dispatch is None:
            raise ValueError(
                f"The tape holds consumer opcode {int(unknown[0])} but no dispatch "
                "kernel was given. Pass dispatch= to compile_tree."
            )
    run, idle, many, many_parallel = _batch_kernels()
    if dispatch is None:
        dispatch = idle
    packed = _pack(tapes)
    use_parallel = parallel and numba.get_num_threads() > 1
    if use_parallel:
        n_threads = numba.get_num_threads()
        stacks = numpy.empty((n_threads, int(packed["max_depth"]) + 1, rows), dtype=numpy.float64)
        scratches = numpy.empty((n_threads, rows), dtype=numpy.float64)
        many_parallel(
            run,
            packed["opcodes"],
            packed["operands"],
            packed["constants"],
            packed["op_starts"],
            packed["op_lens"],
            packed["c_starts"],
            packed["c_lens"],
            packed["fills"],
            matrix,
            stacks,
            scratches,
            dispatch,
            out,
        )
        return out
    stack, scratch = reserve(int(packed["max_depth"]), rows)
    many(
        run,
        packed["opcodes"],
        packed["operands"],
        packed["constants"],
        packed["op_starts"],
        packed["op_lens"],
        packed["c_starts"],
        packed["c_lens"],
        packed["fills"],
        matrix,
        stack,
        scratch,
        dispatch,
        out,
    )
    return out

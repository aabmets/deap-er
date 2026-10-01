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
import threading
from collections.abc import Sequence
from typing import Any

import numba  # optional extra; this module is imported only for backend='numba'
import numpy

from ..opcodes import USER_BASE
from ..tape import Tape, check_tape
from . import numba_kernels
from .numba_compile import build, runner
from .numba_pack import pack_tapes

__all__: list[str] = ["batch_kernel", "compiled_batch_kernels", "launch_kernels", "run_tapes"]

_NO_COLUMNS = (
    "The numba backend evaluates a tape over columns and cannot size a "
    "result without them. Use backend='python' or backend='opcode' for a "
    "primitive set that takes no arguments."
)

_batch: dict[str, Any] = {}
_lock = threading.Lock()


def compiled_batch_kernels() -> frozenset[str]:
    """Names of batch kernels compiled in this process.

    Returns:
        ``many``, ``many_parallel``, ``many_dispatch``, and/or
        ``many_dispatch_parallel`` once each kernel has been built.
    """
    return frozenset(_batch)


def interpret_builtins_parallel(  # pragma: no cover
    streams: Any, layout: Any, columns: Any, stacks: Any, scratches: Any, out: Any
) -> None:
    """Run many builtin-only tapes with one workspace per thread.

    Args:
        streams: ``(opcodes, operands, constants)`` concatenated tapes.
        layout: ``(op_starts, op_lens, c_starts, c_lens, fills)``.
        columns: Packed input matrix.
        stacks: Per-thread workspaces.
        scratches: Per-thread scratch rows.
        out: Result of shape ``(n_tapes, n_rows)``.
    """
    for index in numba.prange(out.shape[0]):  # ty: ignore[not-iterable]
        slot = numba.get_thread_id()
        opcodes, operands, constants, fill = numba_kernels.tape_slices(index, streams, layout)
        numba_kernels.interpret(
            opcodes, operands, constants, columns, fill, stacks[slot], scratches[slot]
        )
        out[index] = stacks[slot, 0]


def interpret_many_parallel(  # pragma: no cover
    run: Any, streams: Any, layout: Any, columns: Any, stacks: Any, scratches: Any, out: Any
) -> None:
    """Run many tapes with one workspace per thread.

    Args:
        run: Compiled single-tape interpreter.
        streams: ``(opcodes, operands, constants)`` concatenated tapes.
        layout: ``(op_starts, op_lens, c_starts, c_lens, fills)``.
        columns: Packed input matrix.
        stacks: Per-thread workspaces.
        scratches: Per-thread scratch rows.
        out: Result of shape ``(n_tapes, n_rows)``.
    """
    for index in numba.prange(out.shape[0]):  # ty: ignore[not-iterable]
        slot = numba.get_thread_id()
        opcodes, operands, constants, fill = numba_kernels.tape_slices(index, streams, layout)
        run(opcodes, operands, constants, columns, fill, stacks[slot], scratches[slot])
        out[index] = stacks[slot, 0]


def batch_kernel(parallel: bool, consumer: bool) -> Any:
    """Compile one batch interpreter once per process.

    The builtin-only kernels call the cached interpreter as a global,
    so they load from the disk cache too. The consumer kernels take a
    closure-bound interpreter as an argument, which Numba cannot cache.

    Args:
        parallel: If True, the ``prange`` kernel.
        consumer: If True, the kernel for tapes with consumer opcodes.

    Returns:
        The compiled batch kernel.
    """
    name = ("many_dispatch" if consumer else "many") + ("_parallel" if parallel else "")
    with _lock:
        if name not in _batch:
            build()
            if consumer:
                kernel = interpret_many_parallel if parallel else numba_kernels.interpret_many
            else:
                kernel = (
                    interpret_builtins_parallel if parallel else numba_kernels.interpret_builtins
                )
            jit = numba.njit(cache=not consumer, nogil=True, error_model="numpy", parallel=parallel)
            _batch[name] = jit(kernel)
        return _batch[name]


def run_tapes(
    tapes: Sequence[Tape],
    matrix: numpy.ndarray,
    dispatch: Any = None,
    parallel: bool = False,
) -> numpy.ndarray:
    """Evaluate tapes on the compiled interpreter.

    Every batch runs a compiled batch kernel: the serial one, or the
    ``prange`` one when ``parallel=True`` and more than one Numba
    thread is available. Neither shares subexpressions across tapes,
    and their floating-point rounding differs from the NumPy CSE plan
    of the ``'opcode'`` backend. NaN patterns match, but values agree
    only to a tolerance, not bit for bit: about ``rtol=1e-9`` and
    ``atol=1e-12``. Running window sums are compensated and start over
    when a window holds no finite sample, so the gap does not grow
    with the row count, and the rolling sum or mean of a series packed
    after NaN padding at least one window long equals that of the
    series alone. The serial and parallel kernels follow the same per-tape
    code, so they agree to the same tolerance or better. Each call
    allocates its own workspace, so concurrent calls from several
    threads are safe.

    Args:
        tapes: Tapes produced by ``lower_tree``.
        matrix: C-contiguous ``(n_rows, n_columns)`` ``float64`` table.
        dispatch: Consumer kernel, or None for builtin-only tapes.
        parallel: If True, use one workspace per Numba thread when
            more than one thread is available.

    Returns:
        ``(n_tapes, n_rows)`` results.

    Raises:
        ValueError: If a tape has no columns, holds a consumer opcode
            without a dispatcher, or is malformed on a path that runs
            the compiled kernels (see ``check_tape``).
    """
    rows = matrix.shape[0]
    if not tapes:
        return numpy.empty((0, rows), dtype=numpy.float64)
    for tape in tapes:
        if tape.columns == 0:
            raise ValueError(_NO_COLUMNS)
        unknown = tape.opcodes[tape.opcodes >= USER_BASE]
        if unknown.size and dispatch is None:
            raise ValueError(
                f"The tape holds consumer opcode {int(unknown[0])} but no dispatch "
                "kernel was given. Pass dispatch= to interpret_tapes."
            )
    use_parallel = parallel and numba.get_num_threads() > 1
    return launch_kernels(tapes, matrix, dispatch, use_parallel)


def launch_kernels(
    tapes: Sequence[Tape], matrix: numpy.ndarray, dispatch: Any, parallel: bool
) -> numpy.ndarray:
    """Run tapes on the serial or ``prange`` compiled batch kernel.

    Args:
        tapes: Tapes to evaluate. Must not be empty.
        matrix: C-contiguous ``(n_rows, n_columns)`` ``float64`` table.
        dispatch: Consumer kernel, or None for builtin-only tapes.
        parallel: If True, use one workspace per Numba thread.

    Returns:
        ``(n_tapes, n_rows)`` results.

    Raises:
        ValueError: If a tape is malformed (see ``check_tape``).
    """
    for tape in tapes:
        check_tape(tape)
    rows = matrix.shape[0]
    out = numpy.empty((len(tapes), rows), dtype=numpy.float64)
    packed = pack_tapes(tapes)
    streams = (packed["opcodes"], packed["operands"], packed["constants"])
    layout = (
        packed["op_starts"],
        packed["op_lens"],
        packed["c_starts"],
        packed["c_lens"],
        packed["fills"],
    )
    # Every call gets its own workspace, so concurrent calls from
    # several threads cannot overwrite each other's stack rows.
    slots = numba.get_num_threads() if parallel else 1
    stacks = numpy.empty((slots, int(packed["max_depth"]) + 1, rows), dtype=numpy.float64)
    scratches = numpy.empty((slots, rows), dtype=numpy.float64)
    workspace = (stacks, scratches) if parallel else (stacks[0], scratches[0])
    kernel = batch_kernel(parallel, dispatch is not None)
    if dispatch is None:
        kernel(streams, layout, matrix, *workspace, out)
    else:
        kernel(runner(dispatch), streams, layout, matrix, *workspace, out)
    return out

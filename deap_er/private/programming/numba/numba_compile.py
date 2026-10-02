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
from typing import Any

from . import (
    numba_kernels,
    numba_numeric,
    numba_predicate,
    numba_window,
    numba_window_extreme,
    numba_window_pair,
    numba_window_pair_reduce,
    numba_window_pair_roll,
    numba_window_roll,
    numba_window_scan,
    numba_window_std,
    numba_window_sum,
    numba_window_ts,
)
from .numba_cache import ensure_numba_cache_dir

__all__: list[str] = ["build", "runner"]

_MISSING = (
    "The numba backend needs the optional 'numba' dependency. "
    "Install it with: pip install deap-er[numba]"
)

_built: dict[Any, Any] = {}
_lock = threading.RLock()


JIT_GROUPS: tuple[tuple[Any, tuple[str, ...]], ...] = (
    (
        numba_numeric,
        (
            "truthy",
            "apply_div",
            "apply_log",
            "apply_sqrt",
            "apply_load",
            "apply_arith",
            "apply_unary",
            "apply_numeric",
        ),
    ),
    (
        numba_predicate,
        (
            "apply_eq",
            "apply_gt_lt",
            "apply_ge_le",
            "apply_and_or",
            "apply_not_where",
            "apply_predicate",
        ),
    ),
    (
        numba_window_scan,
        (
            "scan_variance",
            "scan_pair_moments",
        ),
    ),
    (
        numba_window_extreme,
        (
            "fill_nan_rows",
            "drop_outgoing",
            "pop_dominated",
            "ingest_sample",
            "write_extreme",
            "roll_extreme",
            "roll_minmax",
        ),
    ),
    (
        numba_window_std,
        ("roll_std",),
    ),
    (
        numba_window_sum,
        ("fill_lanes", "plan_split", "block_total", "window_total"),
    ),
    (
        numba_window_roll,
        ("window_lanes", "roll_stats"),
    ),
    (
        numba_window,
        (
            "apply_shift",
            "fill_ema",
            "roll_ema",
            "apply_window",
        ),
    ),
    (
        numba_window_pair_reduce,
        ("reduce_pair",),
    ),
    (
        numba_window_pair_roll,
        ("roll_pair_stats",),
    ),
    (
        numba_window_pair,
        ("apply_pair_window",),
    ),
    (
        numba_window_ts,
        ("window_rank", "roll_ts_window", "apply_ts_window"),
    ),
)


def _compile_group(module: Any, names: tuple[str, ...], jit: Any) -> None:
    for name in names:
        compiled = jit(getattr(module, name))
        setattr(module, name, compiled)
        setattr(numba_kernels, name, compiled)


def _wire_scan() -> None:
    numba_window_std.scan_variance = numba_window_scan.scan_variance
    numba_window_pair_roll.scan_pair_moments = numba_window_scan.scan_pair_moments


def _wire_module(module: Any) -> None:
    if module is numba_numeric:
        numba_predicate.truthy = numba_numeric.truthy
        return
    if module is numba_window_scan:
        _wire_scan()
        return
    if module is numba_window_pair_reduce:
        numba_window_pair_roll.reduce_pair = numba_window_pair_reduce.reduce_pair
        return
    if module is numba_window_extreme:
        numba_window.roll_minmax = numba_window_extreme.roll_minmax
        numba_window_ts.roll_extreme = numba_window_extreme.roll_extreme
        return
    if module is numba_window_std:
        numba_window.roll_std = numba_window_std.roll_std
        return
    if module is numba_window_sum:
        for name in ("fill_lanes", "plan_split", "block_total", "window_total"):
            setattr(numba_window_roll, name, getattr(numba_window_sum, name))
        return
    if module is numba_window_roll:
        numba_window.roll_stats = numba_window_roll.roll_stats
        return
    if module is numba_window_pair_roll:
        numba_window_pair.roll_pair_stats = numba_window_pair_roll.roll_pair_stats


def build() -> Any:
    """Compile the builtin tape interpreter.

    It is built once per process and reused for every tape. Callees
    are compiled first so the interpreter sees compiled globals.

    Returns:
        The interpreter for tapes without consumer opcodes.

    Raises:
        ImportError: If the ``numba`` extra is not installed.
    """
    with _lock:
        if "run" in _built:
            return _built["run"]
        ensure_numba_cache_dir()
        try:
            import numba  # optional extra, imported only when the backend is asked for
        except ImportError as err:
            raise ImportError(_MISSING) from err

        jit = numba.njit(cache=True, nogil=True, error_model="numpy")
        # Numba compiles these from bytecode, so the CPython tracer never
        # sees them. The parity tests exercise every instruction.
        for module, names in JIT_GROUPS:
            _compile_group(module, names, jit)
            _wire_module(module)
        _compile_group(numba_kernels, ("apply_builtin", "tape_slices", "interpret"), jit)
        _built["run"] = numba_kernels.interpret
        return _built["run"]


def runner(dispatch: Any) -> Any:
    """Compile the tape interpreter for one consumer kernel.

    A consumer kernel's numba type differs in every process, so an
    interpreter that took it as an argument would miss the disk cache
    and add another entry to it each time. The kernel is bound by
    closure instead, compiled without the disk cache once per kernel,
    while the builtin interpreter stays cached.

    Args:
        dispatch: Consumer kernel, or None for builtin-only tapes.

    Returns:
        A compiled ``(opcodes, operands, constants, columns, fill,
        stack, scratch)`` interpreter.

    Raises:
        ImportError: If the ``numba`` extra is not installed.
    """
    run = build()
    if dispatch is None:
        return run
    with _lock:
        if dispatch in _built:
            return _built[dispatch]
        import numba  # optional extra, already imported by build()

        interpret = numba.njit(cache=False, nogil=True, error_model="numpy")(
            numba_kernels.interpret_dispatch
        )

        def bound(opcodes, operands, constants, columns, fill, stack, scratch):  # pragma: no cover
            return interpret(opcodes, operands, constants, columns, fill, stack, scratch, dispatch)

        _built[dispatch] = numba.njit(cache=False, nogil=True, error_model="numpy")(bound)
        return _built[dispatch]

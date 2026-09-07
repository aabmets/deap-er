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
    numba_window_ts,
)

__all__: list[str] = ["build"]

_MISSING = (
    "The numba backend needs the optional 'numba' dependency. "
    "Install it with: pip install deap-er[numba]"
)

_built: dict[str, Any] = {}

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
            "row_offset",
            "variance_untrusted",
            "scan_sums",
            "scan_variance",
            "scan_pair_sums",
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
        numba_window_roll,
        ("absorb_stat", "window_totals", "reduce_stats", "std_window", "roll_stats"),
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
        ("absorb_pair", "roll_pair_stats"),
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
    for target in (numba_window_roll, numba_window_pair_roll):
        target.row_offset = numba_window_scan.row_offset
        target.variance_untrusted = numba_window_scan.variance_untrusted
    numba_window_roll.scan_sums = numba_window_scan.scan_sums
    numba_window_roll.scan_variance = numba_window_scan.scan_variance
    numba_window_pair_roll.scan_pair_sums = numba_window_scan.scan_pair_sums
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
    if module is numba_window_roll:
        numba_window.roll_stats = numba_window_roll.roll_stats
        return
    if module is numba_window_pair_roll:
        numba_window_pair.roll_pair_stats = numba_window_pair_roll.roll_pair_stats


def build() -> tuple[Any, Any]:
    """Compile the tape interpreter and the fallback dispatcher.

    Both are built once per process and reused for every tape. Callees
    are compiled first so the interpreter sees compiled globals.

    Returns:
        The interpreter and the no-op dispatcher.

    Raises:
        ImportError: If the ``numba`` extra is not installed.
    """
    if "run" in _built:
        return _built["run"], _built["idle"]
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
    _built["run"] = jit(numba_kernels.interpret)
    _built["idle"] = jit(numba_kernels.idle)
    return _built["run"], _built["idle"]

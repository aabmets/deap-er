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

from collections.abc import Sequence
from typing import Any

import numpy

from deap_er.private.various.case_errors import case_errors

from .columnar import make_column_pset
from .numpy.numpy_ops import add_numpy_primitives
from .primitives.primitive_set_typed import PrimitiveSetTyped
from .tape import Tape
from .tape_batch import interpret_tapes
from .tape_lower import lower_tree
from .window_ops import add_window_ephemeral, add_window_primitives
from .window_pair import add_pair_window_primitives
from .window_ts import add_ts_primitives

__all__: list[str] = ["columnar_pset", "evaluate_columnar"]

_DEFAULT_EMPTY = 1.0e6


def columnar_pset(
    names: Sequence[str],
    *,
    name: str = "MAIN",
    window: tuple[int, int] | None = (2, 64),
    window_name: str = "window",
    pair_windows: bool = False,
    ts: bool = False,
) -> PrimitiveSetTyped:
    """Build a columnar primitive set with the usual kits registered.

    This is ``make_column_pset`` plus the vectorized and causal-window
    kits. Pair-window and time-series primitives are opt-in. A window
    ephemeral is added when ``window`` is a bound pair.

    Args:
        names: Column names, in the order compiled trees receive them.
        name: Name of the primitive set.
        window: Inclusive ``(low, high)`` window-ephemeral bounds.
            ``None`` skips the ephemeral.
        window_name: Ephemeral type name. Must be unique for a given
            bound pair across the process.
        pair_windows: If True, also register ``rolling_corr``,
            ``rolling_cov``, and ``rolling_beta``.
        ts: If True, also register ``ts_rank``, ``ts_argmax``, and
            ``ts_argmin``.

    Returns:
        A typed primitive set ready for ``register_gp``.

    Raises:
        ValueError: If ``names`` is unusable or the window bounds
            are invalid.
    """
    pset = make_column_pset(names, name=name)
    add_numpy_primitives(pset)
    add_window_primitives(pset)
    if pair_windows:
        add_pair_window_primitives(pset)
    if ts:
        add_ts_primitives(pset)
    if window is not None:
        add_window_ephemeral(pset, window_name, window[0], window[1])
    return pset


def evaluate_columnar(
    individuals: Sequence[Any],
    pset: PrimitiveSetTyped,
    matrix: numpy.ndarray,
    target: numpy.ndarray,
    *,
    cases: Sequence[tuple[int, int]] | numpy.ndarray | None = None,
    backend: str = "opcode",
    parallel: bool = False,
    empty: float = _DEFAULT_EMPTY,
    min_valid: int | None = None,
    reduce: bool = True,
) -> list[tuple[float, ...]]:
    """Score trees against one packed column matrix.

    Unique programs are lowered once by ``str(tree)``. The batch is
    run through ``interpret_tapes``. Warmup ``nan`` samples are
    dropped from the MSE, matching the columnar fitness contract.
    Register the result as ``toolbox.evaluate_batch``.

    Args:
        individuals: Trees to score. An empty sequence returns ``[]``.
        pset: Primitive set used to lower each unique tree.
        matrix: Packed ``(n_rows, n_columns)`` column table.
        target: One-dimensional target series, length ``n_rows``.
        cases: Optional half-open ``(start, stop)`` ranges or a
            boolean mask forwarded to ``case_errors``.
        backend: Tape backend, ``'opcode'`` or ``'numba'``.
        parallel: If True, run the Numba path with one workspace
            per thread.
        empty: Fitness used when a prediction has too few finite
            samples, when ``cases`` is empty, or when every case
            is empty.
        min_valid: Minimum finite overlap required when ``cases``
            is omitted. Defaults to half the target length.
        reduce: When ``cases`` is set, return the mean of the case
            errors as a one-objective tuple, including a non-finite
            empty-case value. ``False`` returns the per-case tuple
            for lexicase.

    Returns:
        One fitness tuple per individual, in input order.

    Raises:
        ValueError: If ``target`` is not a one-dimensional series
            whose length matches ``matrix`` rows.
    """
    expected = numpy.asarray(target, dtype=numpy.float64)
    if expected.ndim != 1:
        raise ValueError("target must be a one-dimensional series")
    packed = numpy.asarray(matrix, dtype=numpy.float64)
    if packed.ndim != 2 or packed.shape[0] != expected.shape[0]:
        raise ValueError("matrix rows must match the target length")
    if not individuals:
        return []
    tapes, index = _lower_unique(individuals, pset)
    predicted = interpret_tapes(tapes, packed, backend=backend, parallel=parallel)
    floor = expected.size // 2 if min_valid is None else min_valid
    return [
        _score_prediction(predicted[slot], expected, cases, empty, floor, reduce) for slot in index
    ]


def _lower_unique(
    individuals: Sequence[Any], pset: PrimitiveSetTyped
) -> tuple[list[Tape], list[int]]:
    unique: dict[str, int] = {}
    tapes: list[Tape] = []
    index: list[int] = []
    for individual in individuals:
        key = str(individual)
        slot = unique.get(key)
        if slot is None:
            unique[key] = slot = len(tapes)
            tapes.append(lower_tree(individual, pset))
        index.append(slot)
    return tapes, index


def _score_prediction(
    predicted: numpy.ndarray,
    target: numpy.ndarray,
    cases: Sequence[tuple[int, int]] | numpy.ndarray | None,
    empty: float,
    min_valid: int,
    reduce: bool,
) -> tuple[float, ...]:
    if cases is not None:
        errors = case_errors(predicted, target, cases, empty=empty)
        if not errors:
            return (empty,) if reduce else errors
        if not reduce:
            return errors
        return (float(numpy.mean(errors)),)
    valid = numpy.isfinite(predicted) & numpy.isfinite(target)
    if int(valid.sum()) < min_valid:
        return (empty,)
    diff = predicted[valid] - target[valid]
    return (float(numpy.mean(diff * diff)),)

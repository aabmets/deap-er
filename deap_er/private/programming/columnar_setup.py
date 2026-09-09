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
from .tape_interval import bounds_from_matrix, tape_skip_score
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
    static_filter: bool = True,
) -> list[tuple[float, ...]]:
    """Score trees against one packed column matrix.

    Unique programs are lowered once by ``str(tree)``. The batch is
    run through ``interpret_tapes``. Warmup ``nan`` samples are
    dropped from the MSE, matching the columnar fitness contract.
    Register the result as ``toolbox.evaluate_batch``. When
    ``static_filter`` is true, ``tape_flags`` may skip
    ``interpret_tapes`` for identically ``nan``, constant, or
    warmup-hiding programs and write ``empty`` instead. That path is
    what ``evaluate_invalid`` uses when ``evaluate_batch`` is this
    helper; ``nevals`` still counts the assignment.

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
        static_filter: When true, skip ``interpret_tapes`` for
            tapes that fail the static certificates from
            ``tape_flags``.

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
    floor = expected.size // 2 if min_valid is None else min_valid
    n_rows = int(packed.shape[0])
    if static_filter:
        bounds = bounds_from_matrix(packed)
        skip = [tape_skip_score(tape, bounds, n_rows=n_rows) for tape in tapes]
        score_tapes = [tape for tape, bad in zip(tapes, skip, strict=True) if not bad]
        predicted = (
            interpret_tapes(score_tapes, packed, backend=backend, parallel=parallel)
            if score_tapes
            else numpy.empty((0, n_rows), dtype=numpy.float64)
        )
        remap = _score_slot_map(skip)
        nan_row = numpy.full(n_rows, numpy.nan, dtype=numpy.float64)
        return [
            _score_prediction(
                nan_row if skip[slot] else predicted[remap[slot]],
                expected,
                cases,
                empty,
                floor,
                reduce,
            )
            for slot in index
        ]
    predicted = interpret_tapes(tapes, packed, backend=backend, parallel=parallel)
    return [
        _score_prediction(predicted[slot], expected, cases, empty, floor, reduce) for slot in index
    ]


def _score_slot_map(skip: list[bool]) -> list[int]:
    remap = [-1] * len(skip)
    score_index = 0
    for slot, bad in enumerate(skip):
        if not bad:
            remap[slot] = score_index
            score_index += 1
    return remap


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

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

from .opcode_set import Opcode
from .tape import Tape
from .tape_batch import interpret_tapes
from .tape_lookback import tape_lookback

__all__: list[str] = ["suffix_rescore"]

_SHORT_LOOKBACK = (
    "lookback {lookback} is smaller than the tape bound {bound}. "
    "Refusing to rescore a suffix that would miss history."
)


def suffix_rescore(
    tapes: Iterable[Tape],
    matrix: Any,
    prefix: numpy.ndarray,
    *,
    n_new: int | None = None,
    lookback: int | None = None,
    backend: str = "opcode",
    dispatch: Any = None,
    parallel: bool = False,
) -> numpy.ndarray:
    """Rescore ``lookback + n_new`` trailing rows onto a cached prefix.

    After an append-only ``vstack``, pass the grown packed matrix and
    the scores from before the append. The helper keeps those prefix
    columns (new rows are the present) and writes only the last
    ``n_new`` outputs of a suffix ``interpret_tapes`` call. The full
    series matches a one-shot ``interpret_tapes`` on ``matrix``,
    including warmup ``nan`` s.

    A lookback smaller than :func:`tape_lookback` is rejected so a
    short suffix cannot silently drop history. ``ema`` is IIR: the
    helper scores the full pack so the recurrence matches the oracle.

    Args:
        tapes: Tapes in output-row order. A one-shot iterable is
            consumed once.
        matrix: Grown packed ``(n_rows, n_columns)`` table.
        prefix: Cached scores of shape ``(n_tapes, n_prefix_rows)``.
        n_new: Rows appended after ``prefix``. Inferred from the
            matrix and prefix when omitted.
        lookback: History rows to keep in front of the new block.
            Defaults to the max :func:`tape_lookback` of ``tapes``.
        backend: Forwarded to :func:`interpret_tapes`.
        dispatch: Forwarded to :func:`interpret_tapes`.
        parallel: Forwarded to :func:`interpret_tapes`.

    Returns:
        A new ``(n_tapes, n_rows)`` series. It does not alias
        ``prefix`` or ``matrix``.

    Raises:
        ValueError: If ``n_new`` or ``lookback`` is invalid, if
            ``prefix`` does not line up with the tapes and matrix, or
            if ``lookback`` is below a tape's bound.
    """
    tapes = tuple(tapes)
    packed = _as_grown_matrix(matrix)
    rows = packed.shape[0]
    added = _new_rows(n_new, rows, prefix)
    bound = _required_lookback(tapes)
    used = bound if lookback is None else lookback
    if used < 0:
        raise ValueError(f"lookback must be at least 0, got {used}.")
    if used < bound:
        raise ValueError(_SHORT_LOOKBACK.format(lookback=used, bound=bound))
    _check_prefix(prefix, len(tapes), rows - added)
    span = rows if _has_ema(tapes) else min(rows, used + added)
    scored = interpret_tapes(
        tapes, packed[-span:], backend=backend, dispatch=dispatch, parallel=parallel
    )
    out = numpy.empty((len(tapes), rows), dtype=numpy.float64)
    kept = rows - added
    if kept:
        out[:, :kept] = numpy.asarray(prefix, dtype=numpy.float64)[:, :kept]
    if added:
        out[:, kept:] = scored[:, -added:]
    return out


def _as_grown_matrix(matrix: Any) -> numpy.ndarray:
    """Pack the grown column table as C-contiguous ``float64``."""
    if isinstance(matrix, (list, tuple)):
        raise ValueError(
            "suffix_rescore expects a packed (n_rows, n_columns) matrix, "
            "not a sequence of columns."
        )
    packed = numpy.asarray(matrix)
    if packed.ndim != 2:
        raise ValueError(
            f"suffix_rescore expects a packed (n_rows, n_columns) matrix, "
            f"got ndim={packed.ndim}."
        )
    return numpy.ascontiguousarray(packed, dtype=numpy.float64)


def _new_rows(n_new: int | None, rows: int, prefix: numpy.ndarray) -> int:
    """Infer or validate how many rows were appended."""
    cached = 0 if numpy.asarray(prefix).ndim != 2 else int(numpy.asarray(prefix).shape[1])
    added = rows - cached if n_new is None else n_new
    if added < 0:
        raise ValueError(f"n_new must be at least 0, got {added}.")
    if added > rows:
        raise ValueError(f"n_new {added} exceeds the matrix length {rows}.")
    return added


def _required_lookback(tapes: tuple[Tape, ...]) -> int:
    """Return the max lookback bound across ``tapes``."""
    if not tapes:
        return 0
    return max(tape_lookback(tape) for tape in tapes)


def _check_prefix(prefix: numpy.ndarray, n_tapes: int, kept: int) -> None:
    """Reject a prefix cache that does not line up with the rescore."""
    cached = numpy.asarray(prefix)
    if cached.ndim != 2:
        raise ValueError(
            f"prefix must be a (n_tapes, n_prefix_rows) pack, got ndim={cached.ndim}."
        )
    if cached.shape[0] != n_tapes:
        raise ValueError(f"prefix has {cached.shape[0]} tapes, expected {n_tapes}.")
    if cached.shape[1] < kept:
        raise ValueError(
            f"prefix has {cached.shape[1]} rows, need at least {kept} cached scores."
        )


def _has_ema(tapes: tuple[Tape, ...]) -> bool:
    """Return whether any tape holds the IIR ``ema`` opcode."""
    code = int(Opcode.EMA)
    return any(code in tape.opcodes for tape in tapes)

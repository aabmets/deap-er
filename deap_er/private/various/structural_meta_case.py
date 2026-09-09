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

from deap_er.private.programming.opcodes import Opcode, lower_tree
from deap_er.private.programming.primitives.primitive_set_typed import PrimitiveSetTyped
from deap_er.private.programming.primitives.primitive_tree import PrimitiveTree
from deap_er.private.programming.promote_store import promoted_names

__all__: list[str] = [
    "STRUCTURAL_META_CASES",
    "structural_meta_case_columns",
    "structural_meta_case_weights",
]

STRUCTURAL_META_CASES: tuple[str, ...] = (
    "size",
    "depth",
    "unique_opcodes",
    "promote_hits",
    "non_finite_fraction",
)

_SKIP_OPCODES = frozenset({int(Opcode.COL_LOAD), int(Opcode.CONST)})
_NEED_PRIM_SET = frozenset({"unique_opcodes", "promote_hits"})


def _resolve_columns(columns: Sequence[str] | None) -> tuple[str, ...]:
    if columns is None:
        return STRUCTURAL_META_CASES
    unknown = [name for name in columns if name not in STRUCTURAL_META_CASES]
    if unknown:
        raise ValueError(f"unknown structural columns: {unknown}")
    return tuple(columns)


def _as_tree(individual: Any) -> PrimitiveTree:
    if isinstance(individual, PrimitiveTree):
        return individual
    return PrimitiveTree(individual)


def structural_meta_case_weights(
    columns: Sequence[str] | None = None,
) -> tuple[float, ...]:
    """Return default lexicase signs for structural meta-case columns.

    Bloat metrics default to minimize (-1). ``non_finite_fraction``
    defaults to maximize (+1) so lexicase prefers programs that are
    not finite on every bar.

    Args:
        columns: Subset of :data:`STRUCTURAL_META_CASES`. All
            columns are used when omitted.

    Returns:
        One maximize/minimize sign per column.

    Raises:
        ValueError: If a column name is unknown.
    """
    names = _resolve_columns(columns)
    return tuple(1.0 if name == "non_finite_fraction" else -1.0 for name in names)


def _non_finite_fraction(row: numpy.ndarray) -> float:
    series = numpy.asarray(row, dtype=numpy.float64)
    if series.ndim != 1:
        raise ValueError("each predicted row must be one-dimensional")
    if series.size == 0:
        return numpy.nan
    return float(numpy.mean(~numpy.isfinite(series)))


def _unique_opcode_count(tape: Any) -> int:
    return len({int(value) for value in tape.opcodes if int(value) not in _SKIP_OPCODES})


def _promote_hits(tree: PrimitiveTree, promoted: frozenset[str]) -> int:
    if not promoted:
        return 0
    return sum(1 for node in tree if getattr(node, "name", None) in promoted)


def _predicted_rows(
    individuals: Sequence[Any],
    predicted: numpy.ndarray | Sequence[Sequence[float]] | None,
) -> list[numpy.ndarray] | None:
    if predicted is None:
        return None
    packed = numpy.asarray(predicted, dtype=numpy.float64)
    if packed.ndim == 1:
        if len(individuals) != 1:
            raise ValueError("predicted must have one row per individual")
        packed = packed.reshape(1, -1)
    if packed.ndim != 2 or packed.shape[0] != len(individuals):
        raise ValueError("predicted must have shape (n_individuals, n_rows)")
    return [packed[row] for row in range(packed.shape[0])]


def _unique_opcode_scalar(
    individual: Any,
    tree: PrimitiveTree,
    typed_set: PrimitiveSetTyped,
    tape_cache: dict[str, Any],
) -> float:
    key = str(individual)
    tape = tape_cache.get(key)
    if tape is None:
        tape = lower_tree(tree, typed_set)
        tape_cache[key] = tape
    return float(_unique_opcode_count(tape))


def _column_scalar(
    name: str,
    *,
    tree: PrimitiveTree,
    individual: Any,
    typed_set: PrimitiveSetTyped | None,
    promoted: frozenset[str],
    tape_cache: dict[str, Any],
    predicted_rows: list[numpy.ndarray] | None,
    row: int,
) -> float:
    if name == "size":
        return float(len(tree))
    if name == "depth":
        return float(tree.height)
    if name == "unique_opcodes":
        assert typed_set is not None
        return _unique_opcode_scalar(individual, tree, typed_set, tape_cache)
    if name == "promote_hits":
        return float(_promote_hits(tree, promoted))
    if name == "non_finite_fraction":
        if predicted_rows is None:
            return float(numpy.nan)
        return _non_finite_fraction(predicted_rows[row])
    raise ValueError(f"unknown structural column: {name}")


def structural_meta_case_columns(
    individuals: Sequence[Any],
    *,
    prim_set: PrimitiveSetTyped | None = None,
    predicted: numpy.ndarray | Sequence[Sequence[float]] | None = None,
    columns: Sequence[str] | None = None,
) -> numpy.ndarray:
    """Return cheap structural meta-case columns for a population.

    Shape ``(len(individuals), len(columns))``. Tree metrics need no
    ``predicted``. ``non_finite_fraction`` needs one row per
    individual in ``predicted`` (``(n_ind, n_rows)`` or a sequence of
    1-D series). When ``predicted`` is missing, that column is filled
    with ``numpy.nan``.

    Args:
        individuals: Population whose genomes are ``PrimitiveTree`` or
            list-like node sequences.
        prim_set: Primitive set for ``unique_opcodes`` and
            ``promote_hits``. Required when either column is requested.
        predicted: Optional per-individual output series.
        columns: Subset of :data:`STRUCTURAL_META_CASES`. All columns
            are used when omitted.

    Returns:
        Structural scalars with one row per individual.

    Raises:
        ValueError: If ``individuals`` is empty, a column name is
            unknown, ``prim_set`` is missing for opcode or promote
            columns, or ``predicted`` has the wrong shape.
    """
    if not individuals:
        raise ValueError("individuals must be non-empty")
    names = _resolve_columns(columns)
    if _NEED_PRIM_SET.intersection(names) and prim_set is None:
        raise ValueError("prim_set is required for unique_opcodes and promote_hits")

    predicted_rows = (
        _predicted_rows(individuals, predicted) if "non_finite_fraction" in names else None
    )
    promoted = frozenset(promoted_names(prim_set)) if prim_set is not None else frozenset()
    tape_cache: dict[str, Any] = {}
    matrix = numpy.empty((len(individuals), len(names)), dtype=numpy.float64)

    for row, individual in enumerate(individuals):
        tree = _as_tree(individual)
        for col, name in enumerate(names):
            matrix[row, col] = _column_scalar(
                name,
                tree=tree,
                individual=individual,
                typed_set=prim_set,
                promoted=promoted,
                tape_cache=tape_cache,
                predicted_rows=predicted_rows,
                row=row,
            )
    return matrix

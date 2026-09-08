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
from typing import TYPE_CHECKING

import numpy

from deap_er.private.operators.case_exams import score_case_exams
from deap_er.private.programming.promote_store import promoted_names
from deap_er.private.records.archive_common import ArchiveStats
from deap_er.private.records.case_exam import CaseExam
from deap_er.private.records.case_exam_pool import CaseExamPool
from deap_er.private.records.policy_observation import PolicyObservation

if TYPE_CHECKING:
    from deap_er.private.programming.primitives.primitive_set_typed import PrimitiveSetTyped
    from deap_er.private.typedefs import Individual

__all__: list[str] = [
    "policy_exam_scores",
    "policy_observe",
    "policy_promoted_library_size",
    "policy_solve_bits_from_errors",
    "policy_solve_bits_from_fitness",
    "policy_solve_bits_from_semantic_row",
    "policy_unsolved_count",
]

_SOLVE_ATOL = 1e-12


def _reject_raw_arrays(value: object, *, label: str) -> None:
    """Raise when ``value`` is or contains a raw NumPy array."""
    if isinstance(value, numpy.ndarray):
        raise TypeError(f"{label} must not be a raw array; use summary APIs only")
    if isinstance(value, tuple | list):
        for item in value:
            _reject_raw_arrays(item, label=label)


def _validate_solve_bit(bit: int | float | bool) -> None:
    """Reject values that are not explicit ``0/1`` solve-bit encodings."""
    if isinstance(bit, bool):
        return
    if isinstance(bit, int) and not isinstance(bit, bool):
        if bit in (0, 1):
            return
        raise ValueError("solve_bits must contain only 0 and 1")
    value = float(bit)
    if numpy.isclose(value, 0.0, atol=_SOLVE_ATOL) or numpy.isclose(value, 1.0, atol=_SOLVE_ATOL):
        return
    raise ValueError("solve_bits must contain only 0 and 1")


def _coerce_solve_bits(bits: Sequence[int | float | bool]) -> tuple[int, ...]:
    """Normalize solve bits to a fixed ``0/1`` tuple."""
    _reject_raw_arrays(bits, label="solve_bits")
    for bit in bits:
        _validate_solve_bit(bit)
    return tuple(1 if _bit_is_solved(bit) else 0 for bit in bits)


def _unsolved_count(bits: tuple[int, ...]) -> int:
    """Count unsolved cases in an already coerced solve-bit tuple."""
    return sum(1 for bit in bits if bit == 0)


def _bit_is_solved(bit: int | float | bool) -> bool:
    if isinstance(bit, bool):
        return bit
    if isinstance(bit, int) and not isinstance(bit, bool):
        return bit == 1
    return bool(numpy.isclose(float(bit), 1.0, atol=_SOLVE_ATOL) or float(bit) == 1.0)


def policy_solve_bits_from_errors(errors: tuple[float, ...]) -> tuple[int, ...]:
    """Build observation solve bits from :func:`~deap_er.tools.case_errors`.

    Args:
        errors: One MSE per case segment.

    Returns:
        ``1`` when a case error is within ``1e-12`` of zero.
    """
    _reject_raw_arrays(errors, label="errors")
    return tuple(1 if numpy.isclose(error, 0.0, atol=_SOLVE_ATOL) else 0 for error in errors)


def policy_solve_bits_from_fitness(values: tuple[float, ...]) -> tuple[int, ...]:
    """Build observation solve bits from per-case fitness values.

    Uses the same zero threshold as lexicase and
    :func:`~deap_er.tools.score_case_exams`.

    Args:
        values: One fitness value per catalog case.

    Returns:
        ``1`` when a case is solved at zero error.
    """
    return policy_solve_bits_from_errors(values)


def policy_solve_bits_from_semantic_row(row: Sequence[float]) -> tuple[int, ...]:
    """Coerce one row of :func:`~deap_er.gp.semantic_solve_bits` output.

    Args:
        row: One individual's solve-bit row as a plain sequence.

    Returns:
        ``0/1`` tuple suitable for :func:`policy_observe`.
    """
    _reject_raw_arrays(row, label="row")
    return _coerce_solve_bits(row)


def policy_unsolved_count(solve_bits: Sequence[int]) -> int:
    """Count cases not solved in a solve-bit tuple.

    Args:
        solve_bits: Per-case ``0/1`` flags.

    Returns:
        Number of zeros in ``solve_bits``.
    """
    bits = _coerce_solve_bits(solve_bits)
    return _unsolved_count(bits)


def policy_exam_scores(
    exams: Sequence[CaseExam] | CaseExamPool,
    elites: list[Individual],
    *,
    held_out: CaseExam | None = None,
) -> tuple[float, float | None]:
    """Reduce :func:`~deap_er.tools.score_case_exams` to train / held-out scalars.

    Train exams are every exam in ``exams`` when it is a sequence. When
    ``exams`` is a :class:`~deap_er.records.CaseExamPool`, train scores
    use ``pool.exams`` and ``pool.held_out`` unless ``held_out`` overrides
    the pool marker.

    Args:
        exams: Train exams or a pool with an optional held-out exam.
        elites: Evaluated individuals that supply the case pack.
        held_out: Optional held-out exam that overrides a pool marker.

    Returns:
        ``(train_score, held_out_score)`` where ``train_score`` is the
        sum of train-exam difficulties and ``held_out_score`` is the
        held-out difficulty or ``None``.
    """
    pool_held_out = held_out
    train_exams = exams
    if isinstance(exams, CaseExamPool):
        pool_held_out = exams.held_out if held_out is None else held_out
        train_exams = exams.exams
    train_score = float(sum(score_case_exams(train_exams, elites)))
    if pool_held_out is None:
        return train_score, None
    return train_score, float(score_case_exams([pool_held_out], elites)[0])


def policy_promoted_library_size(prim_set: PrimitiveSetTyped) -> int:
    """Return the promoted-library size for observation fields.

    Args:
        prim_set: Primitive set that may hold a promoted library.

    Returns:
        Number of names returned by :func:`~deap_er.gp.promoted_names`.
    """
    return len(promoted_names(prim_set))


def policy_observe(
    *,
    solve_bits: Sequence[int | float | bool],
    train_score: float,
    held_out_score: float | None = None,
    archive: ArchiveStats | None = None,
    nevals: int = 0,
    rows_seen: int = 0,
    promoted_library_size: int = 0,
    fitness_invalid: bool = False,
    last_action_rejected: bool = False,
) -> PolicyObservation:
    """Build the fixed Push policy observation from summary inputs only.

    Accepts reductions from :func:`~deap_er.tools.case_errors`,
    :func:`~deap_er.tools.score_case_exams`,
    :class:`~deap_er.records.ArchiveStats`, promoted-library counters,
    and eval-budget tallies. Raw NumPy packs, column slices, and
    ``matrix[t]`` are rejected at this boundary.

    Args:
        solve_bits: Per-case ``0/1`` flags for the observed individual.
        train_score: Sum of train-exam difficulty scores.
        held_out_score: Held-out exam difficulty, or ``None``.
        archive: Optional archive summary supplying coverage and
            ``qd_score``.
        nevals: Evaluations consumed this step.
        rows_seen: Rows seen in the evaluation matrix so far.
        promoted_library_size: Count of promoted primitive names.
        fitness_invalid: Whether the observed individual lacks valid
            fitness.
        last_action_rejected: Whether the last policy action was
            rejected.

    Returns:
        One :class:`~deap_er.records.PolicyObservation`.

    Raises:
        TypeError: If ``solve_bits`` is or contains a raw array.
        ValueError: If ``solve_bits`` is not a ``0/1`` sequence or a
            counter is negative.
    """
    bits = _coerce_solve_bits(solve_bits)
    if nevals < 0 or rows_seen < 0 or promoted_library_size < 0:
        raise ValueError("nevals, rows_seen, and promoted_library_size must be non-negative")
    coverage = 0.0
    qd_score = 0.0
    if archive is not None:
        coverage = archive.coverage
        qd_score = archive.qd_score
    return PolicyObservation(
        solve_bits=bits,
        unsolved_count=_unsolved_count(bits),
        train_score=float(train_score),
        held_out_score=None if held_out_score is None else float(held_out_score),
        archive_coverage=coverage,
        qd_score=qd_score,
        nevals=nevals,
        rows_seen=rows_seen,
        promoted_library_size=promoted_library_size,
        fitness_invalid=fitness_invalid,
        last_action_rejected=last_action_rejected,
    )

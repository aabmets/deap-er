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

from collections.abc import Sequence, Sized
from typing import Any, cast

from deap_er.private.records.case_exam import CaseExam

__all__: list[str] = ["island_eval_keys"]


def island_eval_keys(
    exams: Sequence[CaseExam],
    *,
    n_cases: int,
    matrix: object | None = None,
    matrices: Sequence[object] | None = None,
) -> tuple[tuple[Any, ...], ...]:
    """Return per-deme keys for ``step_islands(..., eval_keys=)``.

    Keys compare equal when the exam subset and optional matrix
    identity match, so migrants keep fitness only across demes that
    evaluate on the same cases and packed matrix.

    Args:
        exams: One exam per deme.
        n_cases: Fitness-case count from the current pack.
        matrix: Optional matrix shared by every deme.
        matrices: Optional per-deme matrices. Must match ``exams`` in
            length. Ignored when ``matrix`` is given.

    Returns:
        A tuple of hashable keys, one per exam.

    Raises:
        ValueError: If both ``matrix`` and ``matrices`` are given, or
            if ``matrices`` length does not match ``exams``.
    """
    if matrix is not None and matrices is not None:
        raise ValueError("pass either matrix or matrices, not both")
    if matrices is not None and len(matrices) != len(exams):
        raise ValueError("matrices must have one entry per exam")
    matrix_parts: list[tuple[int | None, int] | None]
    if matrix is not None:
        shared = _matrix_part(matrix)
        matrix_parts = [shared] * len(exams)
    elif matrices is not None:
        matrix_parts = [_matrix_part(item) for item in matrices]
    else:
        matrix_parts = [None] * len(exams)
    return tuple((_exam_part(exam, n_cases), matrix_parts[idx]) for idx, exam in enumerate(exams))


def _exam_part(exam: CaseExam, n_cases: int) -> tuple[Any, ...]:
    if exam.mask is not None:
        if exam.mask.shape[0] != n_cases:
            raise ValueError("a boolean mask must match the series length")
        return ("mask", exam.mask.tobytes(), exam.length)
    ranges = tuple(exam.as_ranges(n_cases))
    return ("ranges", ranges, exam.length)


def _matrix_part(matrix: object) -> tuple[int | None, int]:
    if matrix is None:
        return None, 0
    shape = getattr(matrix, "shape", ())
    if shape:
        return id(matrix), int(shape[0])
    try:
        return id(matrix), len(cast(Sized, matrix))
    except TypeError:
        return id(matrix), 0

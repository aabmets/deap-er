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

from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual
from deap_er.private.records.case_exam import CaseExam

from .case_exams import (
    CaseSolved,
    DifficultyMode,
    ExamLike,
    bound_case_exams,
    elite_solve_matrix,
    exam_difficulty,
)
from .sample_informed_cases import sample_informed_cases

__all__: list[str] = ["guard_case_exams"]


def guard_case_exams(
    exams: ExamLike,
    elites: list[Individual],
    *,
    matrix: numpy.ndarray | None = None,
    trust_matrix: bool = False,
    solved: CaseSolved | None = None,
    held_out: CaseExam | None = None,
    min_cases: int = 1,
    mode: DifficultyMode = "unsolved",
    informed: bool = True,
) -> list[CaseExam]:
    """Repair empty exams and all-solved collapse in place.

    An empty exam is replaced by ``last_good``, then ``held_out``, then
    an informed resample of ``min_cases`` when ``informed`` is true. A
    collapsed exam (difficulty ``0`` under ``mode``) unions
    ``held_out`` and, if still collapsed and ``informed``, bumps the
    subset size through ``sample_informed_cases``. Chronological splits
    stay on the caller.

    Args:
        exams: Pool or sequence of exams to repair.
        elites: Evaluated individuals that supply the case pack.
        matrix: Optional ``(n_elites, n_cases)`` pack.
        trust_matrix: When ``True``, ``matrix`` is accepted on shape alone.
        solved: Optional solve predicate. See :func:`score_case_exams`.
        held_out: Caller-marked exam injected on collapse. A pool's
            ``held_out`` is used when this argument is omitted.
        min_cases: Minimum catalog size after a repair. Combined with
            a pool's ``min_cases`` by taking the larger floor.
        mode: Difficulty used to detect collapse. ``hamming`` keeps a
            specialist subset that ``unsolved`` would treat as solved.
        informed: When ``True``, empty or still-collapsed exams may be
            filled with ``sample_informed_cases``.

    Returns:
        The repaired exam list (the pool's live list when a pool is given).

    Raises:
        ValueError: If ``elites`` is empty, ``n_cases`` is 0, or
            ``min_cases`` is less than 1.
    """
    if min_cases < 1:
        raise ValueError("min_cases must be at least 1")
    n_cases, solve = elite_solve_matrix(elites, matrix, trust_matrix, solved)
    if n_cases == 0:
        raise ValueError("every individual must have a valid fitness of the same length")
    items, pool = bound_case_exams(exams, n_cases)
    extra = held_out if held_out is not None else (pool.held_out if pool else None)
    floor = min_cases
    if pool is not None:
        floor = max(floor, pool.min_cases)
    last_good = pool.last_good if pool is not None else None
    for exam in items:
        _repair_case_exam(
            exam,
            elites,
            n_cases,
            solve,
            extra,
            last_good,
            floor,
            matrix,
            trust_matrix,
            solved,
            mode,
            informed,
        )
        if exam.as_cases(n_cases):
            last_good = exam.copy()
            if pool is not None:
                pool.last_good = last_good
    return items


def _repair_case_exam(
    exam: CaseExam,
    elites: list[Individual],
    n_cases: int,
    solve: numpy.ndarray,
    held_out: CaseExam | None,
    last_good: CaseExam | None,
    min_cases: int,
    matrix: numpy.ndarray | None,
    trust_matrix: bool,
    solved: CaseSolved | None,
    mode: DifficultyMode,
    informed: bool,
) -> None:
    cases = exam.as_cases(n_cases)
    if not cases:
        replacement = last_good or held_out
        if replacement is not None:
            exam.assign(replacement.copy())
            cases = exam.as_cases(n_cases)
        if not cases:
            if not informed:
                return
            fill = _informed_case_fill(elites, min_cases, n_cases, matrix, trust_matrix, solved)
            exam.assign(CaseExam.from_cases(fill, n_cases))
            return
    if exam_difficulty(solve, exam.as_cases(n_cases), mode) > 0:
        return
    if held_out is not None:
        exam.assign(_union_case_exams(exam, held_out, n_cases))
        if exam_difficulty(solve, exam.as_cases(n_cases), mode) > 0:
            return
    if not informed:
        return
    bumped = min(n_cases, max(min_cases, len(exam.as_cases(n_cases)) + 1))
    fill = _informed_case_fill(elites, bumped, n_cases, matrix, trust_matrix, solved)
    exam.assign(CaseExam.from_cases(fill, n_cases))


def _informed_case_fill(
    elites: list[Individual],
    count: int,
    n_cases: int,
    matrix: numpy.ndarray | None,
    trust_matrix: bool,
    solved: CaseSolved | None,
) -> list[int]:
    size = min(max(count, 1), n_cases)
    return sample_informed_cases(
        elites,
        size,
        solved=solved,
        matrix=matrix,
        trust_matrix=trust_matrix,
    )


def _union_case_exams(left: CaseExam, right: CaseExam, n_cases: int) -> CaseExam:
    seen: set[int] = set()
    chosen: list[int] = []
    for idx in (*left.as_cases(n_cases), *right.as_cases(n_cases)):
        if idx not in seen and 0 <= idx < n_cases:
            seen.add(idx)
            chosen.append(idx)
    return CaseExam.from_cases(chosen, n_cases)

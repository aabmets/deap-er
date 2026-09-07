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
from deap_er.private.various.rng import rng

from .case_exam_guard import guard_case_exams
from .case_exams import (
    CaseSolved,
    DifficultyMode,
    ExamLike,
    bound_case_exams,
    elite_solve_matrix,
    score_case_exams,
)
from .mut_case_exam import mut_case_mask, mut_case_ranges

__all__: list[str] = ["next_lexicase_cases"]


def next_lexicase_cases(
    exams: ExamLike,
    elites: list[Individual],
    *,
    matrix: numpy.ndarray | None = None,
    trust_matrix: bool = False,
    solved: CaseSolved | None = None,
    case_count: int | None = None,
    informed: bool = True,
    mut_prob: float = 0.2,
    mode: DifficultyMode = "unsolved",
    held_out: CaseExam | None = None,
    min_cases: int = 1,
    length: int | None = None,
) -> list[int]:
    """Vary exams and return the next ``sel_lexicase(..., cases=)`` subset.

    Scores exams on ``elites``, mutates ranges or mask runs, then guards
    empty and collapsed exams. The mutated or guarded winner is the
    ``cases=`` list. ``informed`` only enables
    ``sample_informed_cases`` inside that guard — it does not overwrite
    a healthy winner. The default path and ``informed=False`` both keep
    variation.

    Args:
        exams: Pool or sequence of exams.
        elites: Evaluated individuals that supply the case pack.
        matrix: Optional ``(n_elites, n_cases)`` pack.
        trust_matrix: When ``True``, ``matrix`` is accepted on shape alone.
        solved: Optional solve predicate. See :func:`score_case_exams`.
        case_count: Minimum catalog size forwarded to the guard when a
            repair is needed. Defaults to ``min_cases``.
        informed: When ``True``, empty or still-collapsed exams may be
            filled with ``sample_informed_cases``. A healthy winner is
            never resampled.
        mut_prob: Per-range or per-run mutation probability.
        mode: Difficulty used to pick the exam that feeds lexicase and
            to detect collapse in the guard.
        held_out: Caller-marked exam injected on collapse.
        min_cases: Minimum catalog size after a guard repair.
        length: Bound for range mutation. Defaults to each exam's
            series span or ``n_cases``.

    Returns:
        Case indices for the next lexicase call.

    Raises:
        ValueError: If ``elites`` or ``exams`` is empty, or ``n_cases`` is 0.
    """
    n_cases, _solve = elite_solve_matrix(elites, matrix, trust_matrix, solved)
    items, pool = bound_case_exams(exams, n_cases)
    if not items:
        raise ValueError("exams must be non-empty")
    for exam in items:
        span = exam.mutation_bound(n_cases) if length is None else length
        _vary_exam(exam, span, mut_prob)
    source: ExamLike = pool if pool is not None else items
    floor = min_cases
    if case_count is not None:
        floor = max(floor, int(case_count))
    repaired = guard_case_exams(
        source,
        elites,
        matrix=matrix,
        trust_matrix=trust_matrix,
        solved=solved,
        held_out=held_out,
        min_cases=floor,
        mode=mode,
        informed=informed,
    )
    scores = score_case_exams(
        repaired,
        elites,
        matrix=matrix,
        trust_matrix=trust_matrix,
        solved=solved,
        mode=mode,
    )
    winner = _argmax_ties(scores)
    return repaired[winner].as_cases(n_cases)


def _vary_exam(exam: CaseExam, length: int, mut_prob: float) -> None:
    if exam.mask is not None:
        mut_case_mask(exam.mask, mut_prob=mut_prob)
        return
    if exam.ranges is not None:
        mut_case_ranges(exam.ranges, length=length, mut_prob=mut_prob)


def _argmax_ties(scores: list[int]) -> int:
    best = max(scores)
    ties = [i for i, score in enumerate(scores) if score == best]
    if len(ties) == 1:
        return ties[0]
    return int(rng.choice(ties))

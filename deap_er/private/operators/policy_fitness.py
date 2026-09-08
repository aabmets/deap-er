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
from deap_er.private.records.case_exam_pool import CaseExamPool

from .case_exams import (
    CaseSolved,
    DifficultyMode,
    ExamLike,
    elite_solve_matrix,
    score_case_exams,
)

__all__: list[str] = [
    "guard_policy_fitness_exam",
    "policy_held_out_fitness",
    "resolve_policy_held_out",
]


def resolve_policy_held_out(
    exams: ExamLike,
    *,
    held_out: CaseExam | None = None,
) -> CaseExam:
    """Return the caller-marked held-out exam for policy fitness.

    Train exams in a :class:`~deap_er.records.CaseExamPool` are never
    returned. Chronological meaning and marking stay on the caller.

    Args:
        exams: Train exams, a pool, or a single exam sequence.
        held_out: Optional held-out exam that overrides a pool marker.

    Returns:
        The resolved held-out exam.

    Raises:
        ValueError: If no held-out exam is marked.
    """
    pool_held_out = held_out
    if isinstance(exams, CaseExamPool) and held_out is None:
        pool_held_out = exams.held_out
    if pool_held_out is None:
        raise ValueError("policy fitness requires a caller-marked held_out exam")
    return pool_held_out


def guard_policy_fitness_exam(
    fitness_exam: CaseExam,
    *,
    held_out: CaseExam,
    n_cases: int,
    train_exams: list[CaseExam] | None = None,
    mutated_exam: CaseExam | None = None,
) -> None:
    """Refuse policy fitness that targets a train or mutated exam.

    Policy individuals must be scored only on ``held_out``. Train-exam
    quality belongs in :func:`~deap_er.tools.policy_observe`, not in
    fitness assignment.

    Args:
        fitness_exam: Exam the caller would score for policy fitness.
        held_out: Caller-marked held-out exam.
        n_cases: Catalog length from the current elite pack.
        train_exams: Optional train exams that must not become the
            fitness target.
        mutated_exam: Optional exam the policy action just varied.

    Raises:
        ValueError: If ``fitness_exam`` is not the held-out exam or
            matches a train or freshly mutated exam.
    """
    held_cases = held_out.as_cases(n_cases)
    fitness_cases = fitness_exam.as_cases(n_cases)
    if train_exams is not None:
        for train in train_exams:
            if train is fitness_exam:
                raise ValueError("train-exam quality is an observation, not the policy objective")
            if train.as_cases(n_cases) == fitness_cases:
                raise ValueError("train-exam quality is an observation, not the policy objective")
    if fitness_cases != held_cases:
        raise ValueError("policy fitness must use the caller-marked held_out exam only")
    if mutated_exam is None:
        return
    if mutated_exam is fitness_exam or mutated_exam is held_out:
        raise ValueError("policy fitness must not reward the exam the policy just mutated")
    if mutated_exam.as_cases(n_cases) == fitness_cases:
        raise ValueError("policy fitness must not reward the exam the policy just mutated")


def policy_held_out_fitness(
    elites: list[Individual],
    exams: ExamLike,
    *,
    held_out: CaseExam | None = None,
    fitness_exam: CaseExam | None = None,
    train_exams: list[CaseExam] | None = None,
    mutated_exam: CaseExam | None = None,
    matrix: numpy.ndarray | None = None,
    trust_matrix: bool = False,
    solved: CaseSolved | None = None,
    mode: DifficultyMode = "unsolved",
) -> float:
    """Score policy individuals only on the held-out exam.

    Train-exam difficulty is not part of the objective. Pair with
    :func:`~deap_er.tools.policy_exam_scores` and
    :func:`~deap_er.records.policy_generalization_gap` for
    observations and logbook chapters.

    Args:
        elites: Evaluated tape individuals that supply the case pack.
        exams: Train exams or a pool with a caller-marked ``held_out``.
        held_out: Optional held-out exam that overrides a pool marker.
        fitness_exam: Optional exam the caller would assign fitness on.
            When set, it must match ``held_out``.
        train_exams: Optional train exams checked by
            :func:`guard_policy_fitness_exam`.
        mutated_exam: Optional exam varied by the last policy action.
        matrix: Optional ``(n_elites, n_cases)`` pack.
        trust_matrix: When ``True``, ``matrix`` is accepted on shape alone.
        solved: Optional solve predicate. See :func:`score_case_exams`.
        mode: ``unsolved`` or ``hamming`` difficulty.

    Returns:
        Held-out exam difficulty as the policy fitness scalar.

    Raises:
        ValueError: If no held-out exam is marked or a guard refuses
            the requested fitness exam.
    """
    resolved = resolve_policy_held_out(exams, held_out=held_out)
    n_cases, _ = elite_solve_matrix(elites, matrix, trust_matrix, solved)
    pool_train: list[CaseExam] | None = train_exams
    if pool_train is None and isinstance(exams, CaseExamPool):
        pool_train = exams.exams
    target = fitness_exam or resolved
    guard_policy_fitness_exam(
        target,
        held_out=resolved,
        n_cases=n_cases,
        train_exams=pool_train,
        mutated_exam=mutated_exam,
    )
    return float(
        score_case_exams(
            [resolved],
            elites,
            matrix=matrix,
            trust_matrix=trust_matrix,
            solved=solved,
            mode=mode,
        )[0]
    )

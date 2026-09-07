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

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual
from deap_er.private.records.case_exam import CaseExam
from deap_er.private.records.case_exam_pool import CaseExamPool, coerce_case_exam

from .sel_lexicase_matrix import fitness_case_matrix, validate_case_matrix

__all__: list[str] = [
    "CaseSolved",
    "DifficultyMode",
    "ExamLike",
    "bound_case_exams",
    "elite_solve_matrix",
    "exam_difficulty",
    "score_case_exams",
]

type CaseSolved = Callable[[Individual, int], bool]
type ExamLike = CaseExam | CaseExamPool | Sequence[CaseExam | Sequence[int] | numpy.ndarray]
type DifficultyMode = str


def score_case_exams(
    exams: ExamLike,
    elites: list[Individual],
    *,
    matrix: numpy.ndarray | None = None,
    trust_matrix: bool = False,
    solved: CaseSolved | None = None,
    mode: DifficultyMode = "unsolved",
) -> list[int]:
    """Score case subsets on elites by how many cases they still fool.

    A case is solved when its value is within ``1e-12`` of zero, matching
    ``sample_informed_cases``. ``unsolved`` counts selected cases that no
    elite solves. ``hamming`` counts unsolved elite-case pairs.

    Args:
        exams: Exams, a :class:`~deap_er.records.CaseExamPool`, or raw
            masks / catalog index lists.
        elites: Evaluated individuals that supply the case pack.
        matrix: Optional ``(n_elites, n_cases)`` pack. Ignored when
            ``solved`` is not the default zero test.
        trust_matrix: When ``True``, ``matrix`` is accepted on shape alone.
        solved: Optional ``(individual, case) -> bool`` predicate.
        mode: ``unsolved`` or ``hamming``.

    Returns:
        One difficulty score per exam.

    Raises:
        ValueError: If ``elites`` is empty, fitness lengths differ, or
            ``mode`` is unknown.
    """
    n_cases, solve = elite_solve_matrix(elites, matrix, trust_matrix, solved)
    items, _pool = bound_case_exams(exams, n_cases)
    return [exam_difficulty(solve, exam.as_cases(n_cases), mode) for exam in items]


def elite_solve_matrix(
    elites: list[Individual],
    matrix: numpy.ndarray | None,
    trust_matrix: bool,
    solved: CaseSolved | None,
) -> tuple[int, numpy.ndarray]:
    """Pack elite solve bits as ``(n_elites, n_cases)``.

    Args:
        elites: Evaluated individuals that supply the case pack.
        matrix: Optional ``(n_elites, n_cases)`` pack.
        trust_matrix: When ``True``, ``matrix`` is accepted on shape alone.
        solved: Optional solve predicate. The pack is ignored when set.

    Returns:
        ``n_cases`` and a boolean solve matrix.

    Raises:
        ValueError: If ``elites`` is empty or fitness lengths differ.
    """
    if not elites:
        raise ValueError("individuals must be non-empty")
    n_cases = len(elites[0].fitness.values)
    if solved is not None:
        bits = numpy.empty((len(elites), n_cases), dtype=bool)
        for row, individual in enumerate(elites):
            values = individual.fitness.values
            if len(values) != n_cases:
                raise ValueError("every individual must have a valid fitness of the same length")
            for case in range(n_cases):
                bits[row, case] = solved(individual, case)
        return n_cases, bits
    packed = fitness_case_matrix(elites) if matrix is None else matrix
    if matrix is not None:
        validate_case_matrix(packed, elites, trust=trust_matrix)
    return n_cases, numpy.isclose(packed, 0.0, atol=1e-12)


def exam_difficulty(solve: numpy.ndarray, cases: list[int], mode: DifficultyMode) -> int:
    """Score one catalog subset against an elite solve matrix.

    Args:
        solve: Boolean ``(n_elites, n_cases)`` solve bits.
        cases: Selected catalog indices.
        mode: ``unsolved`` or ``hamming``.

    Returns:
        The difficulty of ``cases``.

    Raises:
        ValueError: If ``mode`` is unknown.
    """
    if not cases:
        return 0
    cols = solve[:, cases]
    if mode == "unsolved":
        return int(numpy.count_nonzero(~cols.any(axis=0)))
    if mode == "hamming":
        return int(numpy.count_nonzero(~cols))
    raise ValueError("mode must be 'unsolved' or 'hamming'")


def bound_case_exams(
    exams: ExamLike,
    n_cases: int,
) -> tuple[list[CaseExam], CaseExamPool | None]:
    """Resolve a pool or raw exam inputs to a live ``CaseExam`` list.

    Args:
        exams: A pool, one exam, a mask, or a sequence of those.
        n_cases: Catalog length used to coerce index lists.

    Returns:
        The exam list and the pool when ``exams`` is a pool.
    """
    if isinstance(exams, CaseExamPool):
        return exams.exams, exams
    if isinstance(exams, CaseExam | numpy.ndarray):
        return [coerce_case_exam(exams, n_cases)], None
    return [coerce_case_exam(item, n_cases) for item in exams], None

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

from collections.abc import Iterator, Sequence

import numpy

from deap_er.private.records.case_exam import CaseExam, CaseRanges

__all__: list[str] = ["CaseExamPool", "coerce_case_exam"]


class CaseExamPool:
    """Cheap second population of :class:`CaseExam` subsets.

    Stores exams and an optional caller-marked held-out exam. Scoring,
    mutation, and lexicase feed stay on the operators.
    """

    def __init__(
        self,
        exams: Sequence[CaseExam],
        *,
        held_out: CaseExam | None = None,
        min_cases: int = 1,
    ) -> None:
        """Create a pool of exams.

        Args:
            exams: Initial case subsets.
            held_out: Optional exam injected on empty or all-solved collapse.
            min_cases: Minimum catalog size after a guard repair.

        Raises:
            ValueError: If ``min_cases`` is less than 1.
        """
        if min_cases < 1:
            raise ValueError("min_cases must be at least 1")
        self._exams = list(exams)
        self.held_out = held_out
        self.min_cases = min_cases
        self.last_good: CaseExam | None = None

    @property
    def exams(self) -> list[CaseExam]:
        """Live list of exams in this pool."""
        return self._exams

    def __len__(self) -> int:
        """Return the number of stored exams."""
        return len(self._exams)

    def __iter__(self) -> Iterator[CaseExam]:
        """Iterate over stored exams."""
        return iter(self._exams)

    def __getitem__(self, index: int) -> CaseExam:
        """Return the exam at ``index``."""
        return self._exams[index]


def coerce_case_exam(
    exam: CaseExam | CaseRanges | Sequence[int],
    n_cases: int | None = None,
) -> CaseExam:
    """Wrap ranges, a mask, or catalog indices as a :class:`CaseExam`.

    Args:
        exam: An exam, a range table, a 1-D ``bool`` mask, or case indices.
        n_cases: Required when ``exam`` is a sequence of catalog indices.

    Returns:
        ``exam`` if it is already a :class:`CaseExam`, otherwise a new exam.

    Raises:
        ValueError: If case indices are given without ``n_cases``.
    """
    if isinstance(exam, CaseExam):
        return exam
    if isinstance(exam, numpy.ndarray):
        if exam.dtype == bool:
            return CaseExam(mask=exam)
        return CaseExam(ranges=exam)
    if not exam:
        return CaseExam(ranges=[])
    first = exam[0]
    if isinstance(first, tuple | list | numpy.ndarray) and len(first) == 2:
        return CaseExam(ranges=exam)  # type: ignore[arg-type]
    if n_cases is None:
        raise ValueError("n_cases is required to coerce case indices")
    return CaseExam.from_cases(exam, n_cases)  # type: ignore[arg-type]

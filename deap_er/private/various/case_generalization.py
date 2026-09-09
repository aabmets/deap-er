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
from dataclasses import dataclass
from typing import TYPE_CHECKING

from deap_er.private.operators.downsample_schedule import DownsampleMode, next_downsample_cases
from deap_er.private.operators.sel_lexicase import sel_lexicase
from deap_er.private.operators.sel_lexicase_matrix import fitness_case_matrix
from deap_er.private.records.case_exam import CaseExam
from deap_er.private.records.case_exam_pool import CaseExamPool

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = [
    "CaseGeneralizationRecipe",
    "case_generalization_pool",
    "case_generalization_recipe",
    "held_out_tail",
    "make_lexicase_train_select",
    "train_head",
]


def _validate_fraction(fraction: float) -> None:
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be strictly between 0 and 1")


def _validate_n_cases(n_cases: int) -> None:
    if n_cases < 1:
        raise ValueError("n_cases must be at least 1")


def held_out_tail(n_cases: int, fraction: float = 0.2) -> list[int]:
    """Return catalog indices for the last ``fraction`` of cases.

    Chronological meaning stays on the caller. This helper only picks
    trailing catalog indices.

    Args:
        n_cases: Catalog length.
        fraction: Held-out share in ``(0, 1)``.

    Returns:
        Ascending held-out indices.

    Raises:
        ValueError: If ``n_cases`` or ``fraction`` is invalid.
    """
    _validate_n_cases(n_cases)
    _validate_fraction(fraction)
    held_count = max(1, min(n_cases - 1, round(n_cases * fraction)))
    start = n_cases - held_count
    return list(range(start, n_cases))


def train_head(n_cases: int, fraction: float = 0.2) -> list[int]:
    """Return train catalog indices complementary to :func:`held_out_tail`.

    Args:
        n_cases: Catalog length.
        fraction: Held-out share in ``(0, 1)``.

    Returns:
        Ascending train indices.

    Raises:
        ValueError: If ``n_cases`` or ``fraction`` is invalid.
    """
    held = held_out_tail(n_cases, fraction)
    return [idx for idx in range(n_cases) if idx not in held]


def case_generalization_pool(
    n_cases: int,
    *,
    fraction: float = 0.2,
    held_cases: Sequence[int] | None = None,
) -> CaseExamPool:
    """Build a pool with one train exam and a caller-marked held-out exam.

    Args:
        n_cases: Catalog length.
        fraction: Held-out share when ``held_cases`` is omitted.
        held_cases: Optional explicit held-out catalog indices.

    Returns:
        A pool whose train exam excludes ``held_out``.

    Raises:
        ValueError: If ``n_cases`` or ``fraction`` is invalid.
        IndexError: If a held-out index is out of range.
    """
    _validate_n_cases(n_cases)
    chosen = list(held_cases) if held_cases is not None else held_out_tail(n_cases, fraction)
    if not chosen:
        raise ValueError("held_cases must select at least one case")
    held = CaseExam.from_cases(chosen, n_cases)
    train_indices = [idx for idx in range(n_cases) if idx not in set(chosen)]
    if not train_indices:
        raise ValueError("held_cases must leave at least one train case")
    train = CaseExam.from_cases(train_indices, n_cases)
    return CaseExamPool([train], held_out=held)


def make_lexicase_train_select(
    pool: CaseExamPool,
    n_cases: int,
    *,
    downsample: int | None = None,
    downsample_mode: DownsampleMode = "informed",
) -> Callable[[list[Individual], int], list[Individual]]:
    """Return a ``toolbox.select`` callable that runs lexicase on train cases.

    Held-out catalog indices from ``pool.held_out`` are never passed to
    lexicase. Chronological meaning stays on the caller.

    Args:
        pool: Exam pool with a train exam and optional ``held_out``.
        n_cases: Catalog length for ``CaseExam.as_cases``.
        downsample: When set, cap the active case count each generation
            via :func:`~deap_er.operators.next_downsample_cases`.
        downsample_mode: Downsample mode when ``downsample`` is set.

    Returns:
        A selection callable ``(individuals, sel_count) -> list``.

    Raises:
        ValueError: If the pool has no train exams.
    """
    if not pool.exams:
        raise ValueError("pool must contain at least one train exam")
    train_cases = pool.exams[0].as_cases(n_cases)
    held_set = set(pool.held_out.as_cases(n_cases)) if pool.held_out is not None else set()
    generation = 0

    def select(individuals: list[Individual], sel_count: int) -> list[Individual]:
        nonlocal generation
        matrix = fitness_case_matrix(individuals)
        if downsample is None:
            cases = train_cases
        else:
            cases = next_downsample_cases(
                individuals,
                downsample,
                generation,
                mode=downsample_mode,
            )
            cases = [case for case in cases if case in train_cases]
            if not cases:
                cases = train_cases
        if held_set.intersection(cases):
            raise RuntimeError("held-out cases must not reach lexicase selection")
        chosen = sel_lexicase(individuals, sel_count, cases=cases, matrix=matrix)
        generation += 1
        return chosen

    return select


@dataclass(frozen=True, slots=True)
class CaseGeneralizationRecipe:
    """Wiring for the default case-structured generalization path.

    Attributes:
        pool: Train and held-out exams.
        train_cases: Train catalog indices.
        held_cases: Held-out catalog indices, or ``[]`` when unset.
        n_cases: Catalog length.
    """

    pool: CaseExamPool
    train_cases: tuple[int, ...]
    held_cases: tuple[int, ...]
    n_cases: int

    def make_select(
        self,
        *,
        downsample: int | None = None,
        downsample_mode: DownsampleMode = "informed",
    ) -> Callable[[list[Individual], int], list[Individual]]:
        """Return a lexicase selector on :attr:`train_cases` only."""
        return make_lexicase_train_select(
            self.pool,
            self.n_cases,
            downsample=downsample,
            downsample_mode=downsample_mode,
        )


def case_generalization_recipe(
    n_cases: int,
    *,
    fraction: float = 0.2,
    held_cases: Sequence[int] | None = None,
) -> CaseGeneralizationRecipe:
    """Build the held-out pool and catalog indices for item 41.

    Args:
        n_cases: Catalog length.
        fraction: Held-out share when ``held_cases`` is omitted.
        held_cases: Optional explicit held-out catalog indices.

    Returns:
        A recipe with ``pool``, train indices, and held-out indices.
    """
    pool = case_generalization_pool(n_cases, fraction=fraction, held_cases=held_cases)
    train = tuple(pool.exams[0].as_cases(n_cases))
    held = tuple(pool.held_out.as_cases(n_cases)) if pool.held_out is not None else ()
    return CaseGeneralizationRecipe(pool=pool, train_cases=train, held_cases=held, n_cases=n_cases)

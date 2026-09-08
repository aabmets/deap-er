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

from dataclasses import dataclass
from typing import ClassVar

__all__: list[str] = ["PolicyObservation"]


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    """Fixed Push policy observation record.

    This is not a genome and not a domain metric. It is the only
    typed layout a private policy may read: summaries from case
    errors, exams, archives, promoted-library size, and eval budget
    counters. Raw column packs and ``matrix[t]`` never appear here.

    Attributes:
        solve_bits: Per-case ``0/1`` solve flags for the observed
            individual.
        unsolved_count: Number of cases in ``solve_bits`` that are
            not solved.
        train_score: Sum of train-exam difficulty scores from
            :func:`~deap_er.tools.score_case_exams`.
        held_out_score: Held-out exam difficulty, or ``None`` when
            no held-out exam is marked.
        archive_coverage: MAP-Elites coverage from
            :class:`~deap_er.records.ArchiveStats`.
        qd_score: MAP-Elites quality-diversity total from
            :class:`~deap_er.records.ArchiveStats`.
        nevals: Evaluations consumed this generation or step.
        rows_seen: Rows in the evaluation matrix seen so far.
        promoted_library_size: Count of promoted primitive names.
        fitness_invalid: ``True`` when the observed individual lacks
            valid fitness.
        last_action_rejected: ``True`` when the last policy action
            was rejected by a guard.
    """

    solve_bits: tuple[int, ...]
    unsolved_count: int
    train_score: float
    held_out_score: float | None
    archive_coverage: float
    qd_score: float
    nevals: int
    rows_seen: int
    promoted_library_size: int
    fitness_invalid: bool
    last_action_rejected: bool

    FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "solve_bits",
        "unsolved_count",
        "train_score",
        "held_out_score",
        "archive_coverage",
        "qd_score",
        "nevals",
        "rows_seen",
        "promoted_library_size",
        "fitness_invalid",
        "last_action_rejected",
    )

    def as_tuple(
        self,
    ) -> tuple[
        tuple[int, ...],
        int,
        float,
        float | None,
        float,
        float,
        int,
        int,
        int,
        bool,
        bool,
    ]:
        """Return fields in fixed schema order for Push or linear policies."""
        return (
            self.solve_bits,
            self.unsolved_count,
            self.train_score,
            self.held_out_score,
            self.archive_coverage,
            self.qd_score,
            self.nevals,
            self.rows_seen,
            self.promoted_library_size,
            self.fitness_invalid,
            self.last_action_rejected,
        )

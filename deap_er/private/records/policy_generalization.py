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

from typing import Any

from deap_er.private.records.logbook import Logbook

POLICY_GENERALIZATION_GAP_CHAPTER = "generalization_gap"

__all__: list[str] = [
    "POLICY_GENERALIZATION_GAP_CHAPTER",
    "policy_generalization_gap",
    "record_policy_generalization_gap",
]


def policy_generalization_gap(
    train_score: float,
    held_out_score: float | None,
) -> dict[str, float | None]:
    """Build the generalization-gap Logbook chapter payload.

    Train-exam quality is logged for comparison only. Policy fitness
    stays on ``held_out_score`` via
    :func:`~deap_er.tools.policy_held_out_fitness`.

    Args:
        train_score: Sum of train-exam difficulties from
            :func:`~deap_er.tools.policy_exam_scores`.
        held_out_score: Held-out exam difficulty, or ``None`` when
            no held-out exam is marked.

    Returns:
        Chapter fields ``train``, ``held_out``, and ``gap`` where
        ``gap`` is ``train - held_out`` when held-out is present.
    """
    gap = None
    if held_out_score is not None:
        gap = float(train_score) - float(held_out_score)
    return {
        "train": float(train_score),
        "held_out": None if held_out_score is None else float(held_out_score),
        "gap": gap,
    }


def record_policy_generalization_gap(
    logbook: Logbook,
    *,
    gen: int,
    train_score: float,
    held_out_score: float | None,
    **extra: Any,
) -> None:
    """Append one generation row with a generalization-gap chapter.

    Args:
        logbook: Evolution logbook to update.
        gen: Generation index shared with the parent row.
        train_score: Train-exam difficulty sum for observation.
        held_out_score: Held-out difficulty used for policy fitness.
        **extra: Additional parent-row fields such as ``nevals``.
    """
    chapter = policy_generalization_gap(train_score, held_out_score)
    logbook.record(
        gen=gen,
        **extra,
        **{POLICY_GENERALIZATION_GAP_CHAPTER: chapter},
    )

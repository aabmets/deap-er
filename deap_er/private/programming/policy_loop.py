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
from typing import Any

from deap_er.private.algorithms.policy_action import (
    POLICY_ACTION_SKIP_PROMOTE,
    POLICY_ACTION_SKIP_TUNE,
    PolicyActionResult,
    apply_policy_action,
)
from deap_er.private.operators.policy_action_guard import PolicyActionGuard
from deap_er.private.records.archive_common import ArchiveStats
from deap_er.private.records.policy_observation import PolicyObservation
from deap_er.private.various.policy_observe import policy_observe

__all__: list[str] = [
    "POLICY_LOOP_ACTIONS",
    "PolicyLoopStep",
    "policy_action_from_index",
    "policy_action_index",
    "step_policy_loop",
]


POLICY_LOOP_ACTIONS: tuple[str, ...] = (
    "next_lexicase_cases",
    POLICY_ACTION_SKIP_TUNE,
    "promote_subtree",
    POLICY_ACTION_SKIP_PROMOTE,
    "evaluate_invalid",
    "interpret_tapes",
    "step_islands",
)


def policy_action_index(action: str) -> int:
    """Return the discrete index for a policy-loop action token.

    Args:
        action: One of :data:`POLICY_LOOP_ACTIONS`.

    Returns:
        Zero-based index into :data:`POLICY_LOOP_ACTIONS`.

    Raises:
        ValueError: If ``action`` is not part of the loop vocabulary.
    """
    try:
        return POLICY_LOOP_ACTIONS.index(action)
    except ValueError:
        raise ValueError(f"unknown policy loop action: {action}") from None


def policy_action_from_index(index: int) -> str:
    """Map a discrete index back to a policy-loop action token.

    Args:
        index: Zero-based index into :data:`POLICY_LOOP_ACTIONS`.

    Returns:
        The action token at ``index``.

    Raises:
        IndexError: If ``index`` is out of range.
    """
    return POLICY_LOOP_ACTIONS[index]


@dataclass(frozen=True, slots=True)
class PolicyLoopStep:
    """One private policy observe → decide → apply round-trip.

    Attributes:
        observation: Observation built before the decision.
        action: Action token chosen by the policy.
        result: Outcome from :func:`~deap_er.algorithms.apply_policy_action`.
        next_observation: Observation after the action, with
            ``last_action_rejected`` reflecting guard rejection.
    """

    observation: PolicyObservation
    action: str
    result: PolicyActionResult
    next_observation: PolicyObservation


def step_policy_loop(
    decide: Callable[[PolicyObservation], str],
    *,
    solve_bits: Sequence[int | float | bool],
    train_score: float,
    action_kwargs: dict[str, Any],
    held_out_score: float | None = None,
    archive: ArchiveStats | None = None,
    nevals: int = 0,
    rows_seen: int = 0,
    promoted_library_size: int = 0,
    fitness_invalid: bool = False,
    last_action_rejected: bool = False,
    guard: PolicyActionGuard | None = None,
) -> PolicyLoopStep:
    """Run one private policy step behind the P11–P14 firewall.

    Builds a :class:`~deap_er.records.PolicyObservation`, lets ``decide``
    choose a discrete action token, and dispatches through
    :func:`~deap_er.algorithms.apply_policy_action`. The follow-up
    observation surfaces guard rejection via ``last_action_rejected``.

    Counter fields such as ``nevals`` and ``rows_seen`` are caller-owned.
    This helper does not advance them after an action runs; the caller must
    pass updated values on the next step when an evaluation budget or row
    tally should change.

    Unknown action tokens from ``decide`` are rejected by
    :func:`~deap_er.algorithms.apply_policy_action` without raising. The
    follow-up observation sets ``last_action_rejected`` when that happens.

    Args:
        decide: Callable that maps an observation to an action token.
        solve_bits: Per-case ``0/1`` flags for the observed individual.
        train_score: Sum of train-exam difficulty scores.
        action_kwargs: Arguments forwarded to
            :func:`~deap_er.algorithms.apply_policy_action` for the
            chosen token.
        held_out_score: Held-out exam difficulty, or ``None``.
        archive: Optional archive summary.
        nevals: Evaluations consumed this step.
        rows_seen: Rows seen in the evaluation matrix so far.
        promoted_library_size: Count of promoted primitive names.
        fitness_invalid: Whether the observed individual lacks valid
            fitness.
        last_action_rejected: Whether the previous action was rejected.
        guard: Optional action caps from
            :class:`~deap_er.operators.PolicyActionGuard`.

    Returns:
        A :class:`PolicyLoopStep` with before/after observations and
        the dispatch result.
    """
    observation = policy_observe(
        solve_bits=solve_bits,
        train_score=train_score,
        held_out_score=held_out_score,
        archive=archive,
        nevals=nevals,
        rows_seen=rows_seen,
        promoted_library_size=promoted_library_size,
        fitness_invalid=fitness_invalid,
        last_action_rejected=last_action_rejected,
    )
    action = decide(observation)
    result = apply_policy_action(action, guard=guard, **action_kwargs)
    next_observation = policy_observe(
        solve_bits=solve_bits,
        train_score=train_score,
        held_out_score=held_out_score,
        archive=archive,
        nevals=nevals,
        rows_seen=rows_seen,
        promoted_library_size=promoted_library_size,
        fitness_invalid=fitness_invalid,
        last_action_rejected=result.rejected,
    )
    return PolicyLoopStep(
        observation=observation,
        action=action,
        result=result,
        next_observation=next_observation,
    )

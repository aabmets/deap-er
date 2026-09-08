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
from typing import Literal

from deap_er.private.records.policy_observation import PolicyObservation

__all__: list[str] = [
    "LinearPolicyProgram",
    "PolicyDecisionRule",
    "linear_policy_decide",
]

RuleOp = Literal["eq", "ne", "gt", "ge", "lt", "le", "is_true", "is_false"]
RuleField = Literal[
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
]


@dataclass(frozen=True, slots=True)
class PolicyDecisionRule:
    """One fixed decision-list rule on a summary observation field.

    Attributes:
        field: Observation scalar to read. Use ``solve_bit`` with
            :attr:`solve_bit_index` for per-case bits.
        op: Comparison operator.
        value: Right-hand side for numeric comparisons.
        action: Discrete action token to emit when the rule matches.
        solve_bit_index: Index into ``solve_bits`` when ``field`` is
            ``solve_bit``.
    """

    field: RuleField | Literal["solve_bit"]
    op: RuleOp
    value: int | float | bool = 0
    action: str = ""
    solve_bit_index: int = 0


@dataclass(frozen=True, slots=True)
class LinearPolicyProgram:
    """Fixed decision list evaluated top to bottom.

    Attributes:
        rules: Ordered rules; the first match wins.
        default_action: Action emitted when no rule matches.
    """

    rules: tuple[PolicyDecisionRule, ...]
    default_action: str


def linear_policy_decide(
    program: LinearPolicyProgram,
    observation: PolicyObservation,
) -> str:
    """Evaluate a linear policy program on one observation.

    Args:
        program: Fixed decision list.
        observation: Summary observation from
            :func:`~deap_er.tools.policy_observe`.

    Returns:
        The first matching rule action, or :attr:`LinearPolicyProgram.default_action`.
    """
    for rule in program.rules:
        if _rule_matches(rule, observation):
            return rule.action
    return program.default_action


def _rule_matches(rule: PolicyDecisionRule, observation: PolicyObservation) -> bool:
    left = _read_field(rule, observation)
    if rule.op == "is_true":
        return bool(left)
    if rule.op == "is_false":
        return not bool(left)
    right = rule.value
    if rule.op == "eq":
        return left == right
    if rule.op == "ne":
        return left != right
    if rule.op == "gt":
        return left > right
    if rule.op == "ge":
        return left >= right
    if rule.op == "lt":
        return left < right
    if rule.op == "le":
        return left <= right
    raise ValueError(f"unknown policy rule operator: {rule.op}")


def _read_field(rule: PolicyDecisionRule, observation: PolicyObservation) -> int | float | bool:
    if rule.field == "solve_bit":
        bits = observation.solve_bits
        if rule.solve_bit_index < 0 or rule.solve_bit_index >= len(bits):
            return 0
        return bits[rule.solve_bit_index]
    if rule.field == "unsolved_count":
        return observation.unsolved_count
    if rule.field == "train_score":
        return observation.train_score
    if rule.field == "held_out_score":
        return observation.held_out_score if observation.held_out_score is not None else 0.0
    if rule.field == "archive_coverage":
        return observation.archive_coverage
    if rule.field == "qd_score":
        return observation.qd_score
    if rule.field == "nevals":
        return observation.nevals
    if rule.field == "rows_seen":
        return observation.rows_seen
    if rule.field == "promoted_library_size":
        return observation.promoted_library_size
    if rule.field == "fitness_invalid":
        return observation.fitness_invalid
    if rule.field == "last_action_rejected":
        return observation.last_action_rejected
    raise ValueError(f"unknown policy observation field: {rule.field}")

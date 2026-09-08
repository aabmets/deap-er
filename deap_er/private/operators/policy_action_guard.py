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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

from .case_exams import ExamLike, bound_case_exams, elite_solve_matrix

__all__: list[str] = [
    "PolicyActionGuard",
    "estimate_policy_action_evals",
    "guard_policy_action",
]


@dataclass
class PolicyActionGuard:
    """Hard caps for :func:`~deap_er.algorithms.apply_policy_action`.

    Tracks per-generation promote counts, promote cooldown,
    inner tune generation limits, minimum exam size, and an optional
    evaluation budget. Call :meth:`begin_generation` at the start of
    each outer generation so promote limits reset.

    Attributes:
        max_promotes_per_gen: Maximum ``promote_subtree`` calls per
            generation. ``0`` blocks every promote for that generation.
        max_tune_gen: Maximum inner ``n_gen`` accepted by
            ``tune_ephemerals``.
        min_exam_size: Minimum catalog size required for
            ``next_lexicase_cases``.
        promote_cooldown: Generations that must pass after a promote
            before another promote is allowed.
        n_evals: Optional evaluation budget. When set, actions whose
            estimated cost would exceed the remaining budget are
            rejected.
        nevals_used: Evaluations already charged to this guard.
        generation: Current outer generation index.
        promotes_this_gen: Promotes applied in the current generation.
        last_promote_gen: Generation index of the last promote, or
            ``None`` when no promote has run yet.
    """

    max_promotes_per_gen: int = 1
    max_tune_gen: int = 5
    min_exam_size: int = 1
    promote_cooldown: int = 0
    n_evals: int | None = None
    nevals_used: int = 0
    generation: int = 0
    promotes_this_gen: int = 0
    last_promote_gen: int | None = None

    def __post_init__(self) -> None:
        """Reject negative or zero-valued cap configuration."""
        if self.max_promotes_per_gen < 0:
            raise ValueError("max_promotes_per_gen must be at least 0")
        if self.max_tune_gen < 1:
            raise ValueError("max_tune_gen must be at least 1")
        if self.min_exam_size < 1:
            raise ValueError("min_exam_size must be at least 1")
        if self.promote_cooldown < 0:
            raise ValueError("promote_cooldown must be at least 0")
        if self.n_evals is not None and self.n_evals < 0:
            raise ValueError("n_evals must be at least 0")
        if self.nevals_used < 0:
            raise ValueError("nevals_used must be at least 0")

    def begin_generation(self, generation: int | None = None) -> None:
        """Reset per-generation counters and optionally bump ``generation``.

        Args:
            generation: When given, replaces :attr:`generation`.
        """
        if generation is not None:
            self.generation = generation
        self.promotes_this_gen = 0

    def allows(self, action: str, /, **kwargs: Any) -> bool:
        """Return whether ``action`` may run under the current caps.

        Args:
            action: Policy action token.
            **kwargs: Arguments that would be forwarded to
                :func:`~deap_er.algorithms.apply_policy_action`.

        Returns:
            ``False`` when a cap would be violated; otherwise ``True``.
        """
        if action == "promote_subtree" and not self._promote_allowed():
            return False
        if action == "tune_ephemerals":
            n_gen = int(kwargs.get("n_gen", 5))
            if n_gen > self.max_tune_gen:
                return False
        if action == "next_lexicase_cases" and not _exam_size_allowed(
            self.min_exam_size,
            kwargs,
        ):
            return False
        if self.n_evals is not None:
            cost = estimate_policy_action_evals(action, **kwargs)
            if self.nevals_used + cost > self.n_evals:
                return False
        return True

    def note_applied(self, action: str, /, *, evals: int = 0) -> None:
        """Record a successfully applied action.

        Args:
            action: Policy action token that ran.
            evals: Evaluations to charge against :attr:`n_evals`.
        """
        self.nevals_used += evals
        if action == "promote_subtree":
            self.promotes_this_gen += 1
            self.last_promote_gen = self.generation

    def _promote_allowed(self) -> bool:
        if self.promotes_this_gen >= self.max_promotes_per_gen:
            return False
        if self.last_promote_gen is None or self.promote_cooldown <= 0:
            return True
        return self.generation - self.last_promote_gen >= self.promote_cooldown


def guard_policy_action(action: str, guard: PolicyActionGuard, /, **kwargs: Any) -> bool:
    """Return whether ``action`` is allowed under ``guard``.

    Args:
        action: Policy action token.
        guard: Guard state and caps.
        **kwargs: Arguments that would be forwarded to
            :func:`~deap_er.algorithms.apply_policy_action`.

    Returns:
        ``False`` when a cap would be violated; otherwise ``True``.
    """
    return guard.allows(action, **kwargs)


def estimate_policy_action_evals(action: str, /, **kwargs: Any) -> int:
    """Estimate how many evaluations an action would spend.

    Args:
        action: Policy action token.
        **kwargs: Arguments that would be forwarded to
            :func:`~deap_er.algorithms.apply_policy_action`.

    Returns:
        A conservative evaluation count used for budget checks.
    """
    if action == "tune_ephemerals":
        return _tune_eval_cost(kwargs)
    if action == "evaluate_invalid":
        return _evaluate_invalid_cost(kwargs)
    if action == "step_islands":
        return _step_islands_eval_cost(kwargs)
    return 0


def _tune_eval_cost(kwargs: dict[str, Any]) -> int:
    strategy = kwargs.get("strategy")
    n_gen = int(kwargs.get("n_gen", 5))
    offsprings = getattr(strategy, "offsprings", None)
    if offsprings is None:
        offsprings = getattr(strategy, "lamb", 1)
    return n_gen * int(offsprings)


def _evaluate_invalid_cost(kwargs: dict[str, Any]) -> int:
    individuals = kwargs.get("individuals")
    if individuals is None:
        return 0
    return sum(1 for ind in individuals if not ind.fitness.is_valid())


def _step_islands_eval_cost(kwargs: dict[str, Any]) -> int:
    demes = kwargs.get("demes")
    if demes is None:
        return 0
    total = 0
    for _toolbox, population in demes:
        total += sum(1 for ind in population if not ind.fitness.is_valid())
        total += len(population)
    return total


def _exam_size_allowed(min_exam_size: int, kwargs: dict[str, Any]) -> bool:
    floor = int(kwargs.get("min_cases", 1))
    case_count = kwargs.get("case_count")
    if case_count is not None:
        floor = max(floor, int(case_count))
    if floor < min_exam_size:
        return False
    exams = kwargs.get("exams")
    elites = kwargs.get("elites")
    if exams is None or elites is None:
        return True
    return _exams_meet_floor(
        exams,
        elites,
        min_exam_size,
        matrix=kwargs.get("matrix"),
        trust_matrix=kwargs.get("trust_matrix", False),
        solved=kwargs.get("solved"),
    )


def _exams_meet_floor(
    exams: ExamLike,
    elites: list[Individual],
    min_size: int,
    *,
    matrix: Any,
    trust_matrix: bool,
    solved: Any,
) -> bool:
    n_cases, _solve = elite_solve_matrix(elites, matrix, trust_matrix, solved)
    items, _pool = bound_case_exams(exams, n_cases)
    return all(len(exam.as_cases(n_cases)) >= min_size for exam in items)

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

from collections.abc import Callable
from typing import Any

from deap_er.private.operators.policy_action_guard import (
    PolicyActionGuard,
    estimate_policy_action_evals,
)
from deap_er.private.operators.sel_lexicase import sel_lexicase
from deap_er.private.operators.sel_lexicase_matrix import fitness_case_matrix
from deap_er.private.operators.sel_various import sel_best
from deap_er.private.records.case_exam_pool import CaseExamPool
from deap_er.private.records.policy_generalization import policy_generalization_gap
from deap_er.private.records.policy_observation import PolicyObservation
from deap_er.private.toolbox import Toolbox
from deap_er.private.typedefs import Individual
from deap_er.private.various.policy_observe import (
    policy_exam_scores,
    policy_observe,
    policy_solve_bits_from_fitness,
)

from .policy_action import POLICY_ACTION_SKIP_PROMOTE, PolicyActionResult, apply_policy_action

__all__: list[str] = [
    "initial_policy_cases",
    "policy_elites",
    "policy_row_extra",
    "select_policy_offspring",
    "step_policy_generation",
]

_OWNED_OBSERVE = frozenset(
    {
        "solve_bits",
        "train_score",
        "held_out_score",
        "nevals",
        "fitness_invalid",
        "last_action_rejected",
    }
)


def policy_row_extra(
    action: str,
    exams: CaseExamPool | None,
    train_score: float,
    held_out_score: float | None,
) -> dict[str, Any]:
    """Build the logbook extras for one policy generation.

    Args:
        action: Token chosen this generation.
        exams: Optional exam pool. When set, a generalization-gap
            chapter is included.
        train_score: Train-exam difficulty sum.
        held_out_score: Held-out difficulty, or ``None``.

    Returns:
        Fields merged into the generation row.
    """
    extra: dict[str, Any] = {"action": action}
    if exams is not None:
        extra["generalization_gap"] = policy_generalization_gap(train_score, held_out_score)
    return extra


def initial_policy_cases(
    population: list[Individual],
    exams: CaseExamPool | None,
    n_cases: int | None,
) -> list[int] | None:
    """Return the first train exam as lexicase ``cases=``.

    Args:
        population: Individuals that may already have valid fitness.
        exams: Optional exam pool.
        n_cases: Catalog size. Defaults to a valid fitness length.

    Returns:
        Case indices, or ``None`` when no exam can be expanded.
    """
    if exams is None or not exams.exams:
        return None
    catalog = n_cases
    if catalog is None:
        catalog = _fitness_case_count(population)
    if catalog is None:
        return None
    return exams.exams[0].as_cases(catalog)


def select_policy_offspring(
    toolbox: Toolbox,
    population: list[Individual],
    cases: list[int] | None,
) -> list[Individual]:
    """Select the next generation, using lexicase when cases exist.

    Args:
        toolbox: Toolbox with ``select`` when ``cases`` is empty.
        population: Current individuals.
        cases: Lexicase case indices. Empty or ``None`` uses
            ``toolbox.select``.

    Returns:
        Selected individuals, same length as ``population``.
    """
    if not cases:
        return toolbox.select(population, len(population))
    if not population:
        return []
    return sel_lexicase(
        population,
        len(population),
        cases=cases,
        matrix=fitness_case_matrix(population),
    )


def step_policy_generation(
    decide: Callable[[PolicyObservation], str],
    population: list[Individual],
    *,
    used: int,
    rejected: bool,
    exams: CaseExamPool | None,
    guard: PolicyActionGuard | None,
    elite_count: int,
    extras: dict[str, Any],
    observe: dict[str, Any],
    toolbox: Toolbox,
) -> tuple[str, bool, float, float | None, list[int] | None, int]:
    """Observe, decide, and apply one policy action.

    Args:
        decide: Maps an observation to an action token.
        population: Current individuals.
        used: Evaluations consumed so far.
        rejected: Whether the previous action was rejected.
        exams: Optional exam pool for scores and lexicase actions.
        guard: Optional action guard.
        elite_count: Number of elites used to observe.
        extras: Extra kwargs for ``apply_policy_action``.
        observe: Extra kwargs for ``policy_observe``.
        toolbox: Toolbox forwarded to evaluate / rescore actions.

    Returns:
        Action token, rejection flag, train score, held-out score,
        and new lexicase cases when that action applied, plus the
        evaluations the action spent.
    """
    elites = policy_elites(population, elite_count)
    if not elites:
        return POLICY_ACTION_SKIP_PROMOTE, rejected, 0.0, None, None, 0
    first = elites[0]
    solve_bits = policy_solve_bits_from_fitness(first.fitness.values)
    train_score = 0.0
    held_out_score: float | None = None
    if exams is not None:
        train_score, held_out_score = policy_exam_scores(exams, elites)
    observation = policy_observe(
        solve_bits=solve_bits,
        train_score=train_score,
        held_out_score=held_out_score,
        nevals=used,
        fitness_invalid=False,
        last_action_rejected=rejected,
        **{key: value for key, value in observe.items() if key not in _OWNED_OBSERVE},
    )
    action = decide(observation)
    kwargs: dict[str, Any] = {
        **extras,
        "elites": elites,
        "guard": guard,
        "toolbox": toolbox,
        "individuals": population,
    }
    if action != "interpret_tapes":
        kwargs.pop("matrix", None)
    if exams is not None:
        kwargs["exams"] = exams
        if kwargs.get("held_out") is None and exams.held_out is not None:
            kwargs["held_out"] = exams.held_out
    result = apply_policy_action(action, **kwargs)
    applied_cases: list[int] | None = None
    if result.applied and action == "next_lexicase_cases":
        applied_cases = list(result.value)
    return (
        action,
        result.rejected,
        train_score,
        held_out_score,
        applied_cases,
        _action_evals(action, result, kwargs),
    )


def _action_evals(action: str, result: PolicyActionResult, kwargs: dict[str, Any]) -> int:
    if not result.applied:
        return 0
    if action == "evaluate_invalid" and isinstance(result.value, int):
        return result.value
    estimate_kwargs = {key: value for key, value in kwargs.items() if key != "guard"}
    return estimate_policy_action_evals(action, **estimate_kwargs)


def policy_elites(population: list[Individual], elite_count: int) -> list[Individual]:
    """Return the best evaluated individuals for a policy observation.

    Args:
        population: Current individuals.
        elite_count: Maximum number of elites to keep.

    Returns:
        Evaluated elites, or an empty list when none are valid.
    """
    valid = [individual for individual in population if individual.fitness.is_valid()]
    if not valid:
        return []
    return sel_best(valid, min(elite_count, len(valid)))


def _fitness_case_count(population: list[Individual]) -> int | None:
    for individual in population:
        if individual.fitness.is_valid():
            return len(individual.fitness.values)
    return None

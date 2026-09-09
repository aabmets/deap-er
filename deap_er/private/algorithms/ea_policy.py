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

import time
from collections.abc import Callable, Sequence
from logging import Logger
from typing import Any

from deap_er.private.operators.policy_action_guard import PolicyActionGuard
from deap_er.private.records.case_exam_pool import CaseExamPool
from deap_er.private.records.policy_observation import PolicyObservation
from deap_er.private.toolbox import Toolbox
from deap_er.private.typedefs import EvoAlgoResult, EvoRecords, EvoStats, Individual

from .ea_policy_step import (
    initial_policy_cases,
    policy_row_extra,
    select_policy_offspring,
    step_policy_generation,
)
from .loop import budget_spent, check_n_evals, consume_evals, new_logbook, record_generation
from .policy_action import POLICY_ACTION_SKIP_PROMOTE
from .variation import var_and

__all__: list[str] = ["ea_policy"]


def ea_policy(
    toolbox: Toolbox,
    population: list[Individual],
    decide: Callable[[PolicyObservation], str],
    generations: int,
    cx_prob: float,
    mut_prob: float,
    *,
    exams: CaseExamPool | None = None,
    cases: Sequence[int] | None = None,
    n_cases: int | None = None,
    guard: PolicyActionGuard | None = None,
    elite_count: int = 8,
    action_kwargs: dict[str, Any] | None = None,
    observe_kwargs: dict[str, Any] | None = None,
    hof: EvoRecords | None = None,
    stats: EvoStats | None = None,
    verbose: bool = False,
    logger: Logger | None = None,
    log_time: bool = False,
    fronts: list[Any] | None = None,
    n_evals: int | None = None,
) -> EvoAlgoResult:
    """Evolve a population with one policy step each generation.

    Same survivor rule as ``ea_simple``: evaluate invalids, then each
    generation observe → decide → ``apply_policy_action``, select,
    vary, and evaluate. Fitness stays on the toolbox. When ``cases``
    or a pool of exams is available, selection is lexicase on that
    subset; otherwise ``toolbox.select`` is used.

    Requires ``mate``, ``mutate``, and ``evaluate`` (or
    ``evaluate_batch``) on ``toolbox``. ``select`` is required only
    when no case subset is in play.

    Args:
        toolbox: Toolbox with the evolution operators.
        population: Individuals to evolve. Replaced in place.
        decide: Maps one ``PolicyObservation`` to an action token.
        generations: Number of generations to run.
        cx_prob: Probability of mating two individuals.
        mut_prob: Probability of mutating an individual.
        exams: Optional exam pool. Train / held-out scores become
            observations. ``next_lexicase_cases`` reads this pool.
        cases: Initial lexicase case indices. Defaults to the first
            train exam when ``exams`` is set.
        n_cases: Catalog size for ``CaseExam.as_cases``. Defaults to
            the length of a valid fitness vector.
        guard: Optional action guard. ``begin_generation`` is called
            at the start of each outer generation.
        elite_count: Elites used to build the observation.
        action_kwargs: Extra kwargs forwarded to
            ``apply_policy_action`` (for example ``mut_prob`` or
            ``prim_set``).
        observe_kwargs: Extra kwargs forwarded to ``policy_observe``
            (``archive``, ``rows_seen``, ``promoted_library_size``).
        hof: Optional HallOfFame or ParetoFront to update.
        stats: Optional Statistics or MultiStatistics to compile.
        verbose: If True, print the logbook stream each generation.
        logger: If given with ``verbose``, the stream is logged.
        log_time: If True, record per-generation ``duration``.
        fronts: Optional list that receives a ParetoFront snapshot
            of each generation's population.
        n_evals: Optional evaluation budget. ``None`` keeps the
            generation limit only.

    Returns:
        The final population and the logbook.

    Raises:
        ValueError: If ``n_evals`` is negative.
    """
    check_n_evals(n_evals)
    logbook = new_logbook(stats, log_time=log_time, extra_fields=("action",))
    extras = action_kwargs or {}
    observe = observe_kwargs or {}
    active_cases = list(cases) if cases is not None else None
    rejected = False
    action = POLICY_ACTION_SKIP_PROMOTE
    train_score = 0.0
    held_out_score: float | None = None

    t0 = time.perf_counter()
    nevals, used = consume_evals(toolbox, population, n_evals, 0)
    if active_cases is None:
        active_cases = initial_policy_cases(population, exams, n_cases)
    extra = policy_row_extra(action, exams, train_score, held_out_score)
    _record(
        logbook,
        0,
        nevals,
        population,
        population,
        hof,
        stats,
        verbose,
        logger,
        time.perf_counter() - t0 if log_time else None,
        fronts,
        extra,
    )
    if budget_spent(n_evals, used):
        return population, logbook

    for gen in range(1, generations + 1):
        t0 = time.perf_counter()
        if guard is not None:
            guard.begin_generation()
        action, rejected, train_score, held_out_score, applied_cases = step_policy_generation(
            decide,
            population,
            used=used,
            rejected=rejected,
            exams=exams,
            guard=guard,
            elite_count=elite_count,
            extras=extras,
            observe=observe,
            toolbox=toolbox,
        )
        if applied_cases is not None:
            active_cases = applied_cases
        offspring = select_policy_offspring(toolbox, population, active_cases)
        offspring = var_and(toolbox, offspring, cx_prob, mut_prob)
        nevals, used = consume_evals(toolbox, offspring, n_evals, used)
        population[:] = offspring
        _record(
            logbook,
            gen,
            nevals,
            population,
            offspring,
            hof,
            stats,
            verbose,
            logger,
            time.perf_counter() - t0 if log_time else None,
            fronts,
            policy_row_extra(action, exams, train_score, held_out_score),
        )
        if budget_spent(n_evals, used):
            break

    return population, logbook


def _record(
    logbook: Any,
    gen: int,
    nevals: int,
    population: list[Individual],
    offspring: list[Individual],
    hof: EvoRecords | None,
    stats: EvoStats | None,
    verbose: bool,
    logger: Logger | None,
    duration: float | None,
    fronts: list[Any] | None,
    extra: dict[str, Any],
) -> None:
    record_generation(
        logbook,
        gen,
        nevals,
        population=population,
        offspring=offspring,
        hof=hof,
        stats=stats,
        verbose=verbose,
        logger=logger,
        duration=duration,
        fronts=fronts,
        extra=extra,
    )

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

from deap_er.private.records.case_exam import CaseExam
from deap_er.private.records.case_exam_pool import CaseExamPool

from .memetic import numeric_leaves, tune_ephemerals
from .memetic_defaults import (
    MEMETIC_DEFAULT_N_GEN,
    cap_tune_n_gen,
    estimate_tune_ephemerals_evals,
)

__all__: list[str] = ["resolve_tune_held_out", "tune_ephemerals_budget"]


def resolve_tune_held_out(
    exams: Any,
    *,
    held_out: CaseExam | None = None,
) -> CaseExam | None:
    """Return a caller-marked held-out exam when one is available.

    Args:
        exams: Optional :class:`~deap_er.records.CaseExamPool` or
            other exam input. Only a pool marker is read.
        held_out: Optional held-out exam that overrides a pool marker.

    Returns:
        The resolved held-out exam, or ``None`` when none is marked.
    """
    if held_out is not None:
        return held_out
    if isinstance(exams, CaseExamPool) and exams.held_out is not None:
        return exams.held_out
    return None


def tune_ephemerals_budget(
    individual: Any,
    strategy: Any,
    evaluate: Callable[[Any], Any] | None = None,
    *,
    n_gen: int | None = None,
    evaluate_batch: Callable[[list[Any]], Any] | None = None,
    clone: Callable[[Any], Any] | None = None,
    n_evals: int | None = None,
    nevals_used: int = 0,
    exams: Any = None,
    held_out: CaseExam | None = None,
    held_out_evaluate: Callable[[Any], Any] | None = None,
    held_out_evaluate_batch: Callable[[list[Any]], Any] | None = None,
) -> tuple[Any, int]:
    """Polish numeric leaves under a memetic and evaluation leash.

    Caps inner ``n_gen`` to :data:`~deap_er.gp.MEMETIC_MAX_N_GEN` and,
    when ``n_evals`` is set, to the remaining evaluation budget. Defaults
    to :data:`~deap_er.gp.MEMETIC_DEFAULT_N_GEN` inner generations.

    When a caller-marked held-out exam is available from ``held_out`` or
    ``exams``, trials are judged with ``held_out_evaluate`` or
    ``held_out_evaluate_batch`` instead of the train ``evaluate`` path.

    Args:
        individual: ``PrimitiveTree`` or ``SlimTree`` to tune in place.
        strategy: ``Strategy`` or ``StrategySeparable`` whose ``dim``
            matches the leaf count.
        evaluate: ``callable(ind) ->`` fitness tuple for train scoring.
        n_gen: Requested inner generations. Defaults to
            :data:`~deap_er.gp.MEMETIC_DEFAULT_N_GEN`.
        evaluate_batch: Optional ``callable(inds) ->`` fitness tuples
            for train scoring.
        clone: Individual copier passed through to
            :func:`~deap_er.gp.tune_ephemerals`.
        n_evals: Optional evaluation budget for the outer run.
        nevals_used: Evaluations already charged to the run.
        exams: Optional exam pool whose ``held_out`` marker is read
            when ``held_out`` is omitted.
        held_out: Optional held-out exam that overrides a pool marker.
        held_out_evaluate: Held-out judge when a held-out exam exists.
        held_out_evaluate_batch: Batch held-out judge when a held-out
            exam exists.

    Returns:
        ``(individual, evals_spent)``. ``evals_spent`` is ``0`` when the
        budget is already spent or no numeric leaves exist.

    Raises:
        ValueError: If a held-out exam is marked but no held-out judge
            is given, or if neither train nor held-out evaluation is
            available.
    """
    resolved_held_out = resolve_tune_held_out(exams, held_out=held_out)
    judge_evaluate = evaluate
    judge_batch = evaluate_batch
    if resolved_held_out is not None:
        if held_out_evaluate is None and held_out_evaluate_batch is None:
            raise ValueError("held_out exam requires held_out_evaluate or held_out_evaluate_batch")
        judge_evaluate = held_out_evaluate
        judge_batch = held_out_evaluate_batch
    if judge_evaluate is None and judge_batch is None:
        raise ValueError("Provide evaluate or evaluate_batch.")
    requested = MEMETIC_DEFAULT_N_GEN if n_gen is None else int(n_gen)
    capped = cap_tune_n_gen(
        strategy,
        requested,
        n_evals=n_evals,
        nevals_used=nevals_used,
    )
    if capped < 1 or not numeric_leaves(individual):
        return individual, 0
    tune_ephemerals(
        individual,
        strategy,
        evaluate=judge_evaluate,
        n_gen=capped,
        evaluate_batch=judge_batch,
        clone=clone,
    )
    return individual, estimate_tune_ephemerals_evals(strategy, capped)

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

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = [
    "CaseHalvingResult",
    "case_eval_charge",
    "case_halving_stages",
    "evaluate_case_halving",
    "subset_evaluate_cases",
]


def case_eval_charge(n_individuals: int, n_cases: int) -> int:
    """Return case-eval units for budget accounting.

    Args:
        n_individuals: Individuals scored at one rung.
        n_cases: Cases scored per individual.

    Returns:
        ``n_individuals * n_cases``.

    Raises:
        ValueError: If either count is negative.
    """
    if n_individuals < 0 or n_cases < 0:
        raise ValueError("n_individuals and n_cases must be non-negative")
    return n_individuals * n_cases


def case_halving_stages(
    n_train_cases: int,
    *,
    eta: int = 2,
    min_cases: int = 1,
) -> tuple[int, ...]:
    """Return successive-halving rung sizes up to ``n_train_cases``.

    Each rung uses the first ``stage_n`` entries of the caller's
    ``train_cases`` list. Sizes grow geometrically by ``eta`` from
    ``min_cases`` until the full train count is reached.

    Args:
        n_train_cases: Number of train catalog indices.
        eta: Elimination factor and geometric growth base. Must be
            at least ``2``.
        min_cases: Smallest rung size. Must be at least ``1``.

    Returns:
        Distinct ascending rung sizes ending at ``n_train_cases``.

    Raises:
        ValueError: If inputs are invalid.
    """
    if n_train_cases < 1:
        raise ValueError("n_train_cases must be at least 1")
    if eta < 2:
        raise ValueError("eta must be at least 2")
    if min_cases < 1:
        raise ValueError("min_cases must be at least 1")
    if n_train_cases == 1:
        return (1,)
    floor = min(min_cases, n_train_cases)
    stages: list[int] = []
    current = floor
    while current < n_train_cases:
        stages.append(current)
        next_size = current * eta
        if next_size >= n_train_cases:
            break
        current = next_size
    if not stages or stages[-1] != n_train_cases:
        stages.append(n_train_cases)
    return tuple(sorted(set(stages)))


def subset_evaluate_cases(
    evaluate: Callable[[Individual], Sequence[float]],
) -> Callable[[Individual, Sequence[int]], Sequence[float]]:
    """Wrap a full-catalog evaluate for partial case scoring.

    The wrapped callable still invokes ``evaluate`` on the full catalog.
    For true cheap subsets, pass a custom ``evaluate_cases`` that calls
    ``evaluate_columnar(..., cases=...)`` or an equivalent partial scorer.

    Args:
        evaluate: Full-catalog fitness callable.

    Returns:
        ``(individual, cases) ->`` scores for ``cases`` only.
    """

    def evaluate_cases(individual: Individual, cases: Sequence[int]) -> Sequence[float]:
        values = evaluate(individual)
        return tuple(values[int(case)] for case in cases)

    return evaluate_cases


def _mean_score(
    scores: Sequence[float],
    weights: Sequence[float] | None,
) -> float:
    if not scores:
        return float("inf")
    if weights is None:
        return float(sum(scores) / len(scores))
    total = 0.0
    weight_sum = 0.0
    for value, weight in zip(scores, weights, strict=False):
        total += float(value) * float(weight)
        weight_sum += float(weight)
    if weight_sum == 0.0:
        return float("inf")
    return total / weight_sum


def _rank_on_cases(
    survivors: list[Individual],
    evaluate_cases: Callable[[Individual, Sequence[int]], Sequence[float]],
    subset: Sequence[int],
    weights: Sequence[float] | None,
) -> list[tuple[float, Individual]]:
    ranked: list[tuple[float, Individual]] = []
    stage_weights = None
    if weights is not None:
        stage_weights = [weights[int(case)] for case in subset]
    for individual in survivors:
        scores = evaluate_cases(individual, subset)
        ranked.append((_mean_score(scores, stage_weights), individual))
    ranked.sort(key=lambda item: item[0])
    return ranked


def _keep_top(ranked: list[tuple[float, Individual]], eta: int) -> list[Individual]:
    keep = max(1, len(ranked) // eta)
    return [individual for _, individual in ranked[:keep]]


@dataclass(frozen=True, slots=True)
class CaseHalvingResult:
    """Outcome of :func:`evaluate_case_halving`.

    Attributes:
        survivors: Individuals that reached the final rung.
        nevals: Case-eval units charged across all rungs.
        stages_run: Number of rungs executed.
    """

    survivors: list[Individual]
    nevals: int
    stages_run: int


def evaluate_case_halving(
    individuals: Sequence[Individual],
    evaluate_cases: Callable[[Individual, Sequence[int]], Sequence[float]],
    train_cases: Sequence[int],
    *,
    n_cases: int,
    eta: int = 2,
    min_cases: int = 1,
    weights: Sequence[float] | None = None,
    evaluate_full: Callable[[Individual], Sequence[float]] | None = None,
) -> CaseHalvingResult:
    """Run successive halving on train catalog indices.

    Intermediate rungs rank individuals on prefixes of ``train_cases``
    without writing ``fitness.values``. The final rung assigns a
    full-catalog fitness tuple of length ``n_cases``.

    Args:
        individuals: Candidates to score and filter.
        evaluate_cases: ``(individual, case_indices) ->`` per-case scores.
        train_cases: Train catalog indices in caller order.
        n_cases: Full catalog length for final ``fitness.values``.
        eta: Elimination factor between rungs.
        min_cases: Smallest rung size.
        weights: Optional per-catalog weights for ranking means.
        evaluate_full: Optional full-catalog scorer for the final
            rung. Defaults to ``evaluate_cases(ind, range(n_cases))``.

    Returns:
        Survivors, total case-eval charge, and rung count.

    Raises:
        ValueError: If ``n_cases`` or halving parameters are invalid.
    """
    if n_cases < 1:
        raise ValueError("n_cases must be at least 1")
    catalog = list(train_cases)
    if not catalog or not individuals:
        return CaseHalvingResult([], 0, 0)

    stages = case_halving_stages(len(catalog), eta=eta, min_cases=min_cases)
    survivors = list(individuals)
    charged = 0

    def assign_full(individual: Individual) -> Sequence[float]:
        return evaluate_cases(individual, list(range(n_cases)))

    assign = evaluate_full if evaluate_full is not None else assign_full

    for stage_n in stages[:-1]:
        subset = catalog[:stage_n]
        ranked = _rank_on_cases(survivors, evaluate_cases, subset, weights)
        charged += case_eval_charge(len(ranked), stage_n)
        survivors = _keep_top(ranked, eta)

    final_subset = catalog[: stages[-1]]
    charged += case_eval_charge(len(survivors), len(final_subset))
    for individual in survivors:
        individual.fitness.values = tuple(assign(individual))

    return CaseHalvingResult(survivors, charged, len(stages))

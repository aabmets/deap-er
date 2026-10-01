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

import math
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual
from deap_er.private.various.rng import rng

from .epsilon_lexicase_slack import (
    LexicaseMode,
    apply_epsilon_filter,
    apply_strict_filter,
    epsilon_mode_uses_pool_elite,
    epsilon_mode_uses_pool_mad,
    slack_for_case,
)

__all__: list[str] = ["lexicase_select_vectorized"]


def _choice_from_survivors(
    individuals: list[Individual],
    survivors: numpy.ndarray,
) -> Individual:
    if survivors.size == 0:
        return rng.choice(individuals)
    return rng.choice([individuals[i] for i in survivors])


def lexicase_select_vectorized(
    individuals: list[Individual],
    sel_count: int,
    matrix: numpy.ndarray,
    subset: list[int],
    fit_weights: tuple[float, ...],
    *,
    mode: LexicaseMode = "strict",
    epsilon: float | None = None,
) -> list[Individual]:
    """Select individuals by vectorized lexicase filtering.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.
        matrix: Case matrix with shape ``(len(individuals), n_cases)``.
        subset: Fitness-case indices to filter on.
        fit_weights: Per-case maximize/minimize signs from fitness.
        mode: ``strict``, population MAD (``epsilon_auto`` /
            ``epsilon_static``), semi-dynamic pool elite
            (``epsilon_semi``), dynamic pool MAD and elite
            (``epsilon_dynamic``), or fixed slack (``epsilon_fixed``).
        epsilon: Fixed slack when ``mode`` is ``epsilon_fixed``.

    Returns:
        The selected individuals.

    Raises:
        ValueError: If ``mode`` is ``epsilon_fixed`` and ``epsilon`` is
            missing, negative, or not finite.
    """
    if sel_count <= 0:
        return []
    if mode == "epsilon_fixed":
        if epsilon is None:
            raise ValueError("epsilon must be set for epsilon_fixed mode")
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError(f"epsilon must be a finite number >= 0, got {epsilon}")
    pool_elite = epsilon_mode_uses_pool_elite(mode)
    # Population-MAD slack does not depend on the filter pool, so it is
    # computed at most once per case per call instead of per selection.
    slack_cache: dict[int, float] | None = None if epsilon_mode_uses_pool_mad(mode) else {}
    selected: list[Individual] = []
    for _ in range(sel_count):
        order = list(subset)
        rng.shuffle(order)
        active = _filter_cases(
            len(individuals),
            matrix,
            order,
            fit_weights,
            mode,
            epsilon,
            pool_elite,
            slack_cache,
        )
        survivors = numpy.flatnonzero(active)
        selected.append(_choice_from_survivors(individuals, survivors))
    return selected


def _filter_cases(
    n_individuals: int,
    matrix: numpy.ndarray,
    order: list[int],
    fit_weights: tuple[float, ...],
    mode: LexicaseMode,
    epsilon: float | None,
    pool_elite: bool,
    slack_cache: dict[int, float] | None,
) -> numpy.ndarray:
    active = numpy.ones(n_individuals, dtype=bool)
    for case in order:
        if active.sum() <= 1:
            break
        col = matrix[:, case]
        maximize = fit_weights[case] > 0
        if mode == "strict":
            active = apply_strict_filter(active, col, maximize)
        else:
            slack = _case_slack(col, active, mode, epsilon, case, slack_cache)
            active = apply_epsilon_filter(
                active,
                col,
                maximize,
                slack,
                pool_elite=pool_elite,
            )
    return active


def _case_slack(
    col: numpy.ndarray,
    active: numpy.ndarray,
    mode: LexicaseMode,
    epsilon: float | None,
    case: int,
    slack_cache: dict[int, float] | None,
) -> float:
    if slack_cache is None:
        return slack_for_case(col, active, mode, epsilon)
    slack = slack_cache.get(case)
    if slack is None:
        slack = slack_for_case(col, active, mode, epsilon)
        slack_cache[case] = slack
    return slack

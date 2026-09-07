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
from typing import TYPE_CHECKING, Any

import numpy

from deap_er.private.operators.bounds import broadcast_param
from deap_er.private.various.rng import rng
from deap_er.private.various.sort_non_dominated import sort_non_dominated

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

from .common import apply_box_bounds

__all__ = ["resample_offspring", "clip_offspring"]


def _raw(parent: Individual, sigma: float, big_a: numpy.ndarray, step: numpy.ndarray):
    return numpy.asarray(parent + sigma * numpy.dot(big_a, step), dtype=float)


def _front(parents: list[Individual]) -> list[Individual]:
    if all(ind.fitness.is_valid() for ind in parents):
        return sort_non_dominated(parents, len(parents))[0]
    return parents


def resample_offspring(
    strategy: Any, ind_init: Callable[..., Individual], arz: numpy.ndarray, one_each: bool
) -> list[Individual]:
    """Sample offspring, redrawing any point that falls outside the box.

    Parent-index draws stay interleaved with redraws so the RNG order
    matches the previous per-row path.

    Args:
        strategy: Multi-objective CMA strategy.
        ind_init: Callable that turns a sampled vector into an individual.
        arz: Standard-normal steps, shape ``(offsprings, dim)``.
        one_each: If True, parent ``i`` produces offspring ``i``.

    Returns:
        Newly sampled individuals.
    """
    n_dom: list[Individual] = [] if one_each else _front(strategy.parents)
    individuals = []
    for i in range(strategy.lamb):
        p_idx = i if one_each else n_dom[rng.integers(0, len(n_dom))].ps_[1]
        raw = _raw(strategy.parents[p_idx], strategy.sigmas[p_idx], strategy.big_a[p_idx], arz[i])
        init = ind_init(
            apply_box_bounds(
                raw,
                strategy.low,
                strategy.up,
                "resample",
                strategy.resample_limit,
                lambda p_idx=p_idx: _raw(
                    strategy.parents[p_idx],
                    strategy.sigmas[p_idx],
                    strategy.big_a[p_idx],
                    rng.standard_normal(strategy.dim),
                ),
            )
        )
        init.ps_ = "o", p_idx
        individuals.append(init)
    return individuals


def clip_offspring(
    strategy: Any, ind_init: Callable[..., Individual], arz: numpy.ndarray, one_each: bool
) -> list[Individual]:
    """Sample offspring and clip the batch when any box bound is set.

    Args:
        strategy: Multi-objective CMA strategy.
        ind_init: Callable that turns a sampled vector into an individual.
        arz: Standard-normal steps, shape ``(offsprings, dim)``.
        one_each: If True, parent ``i`` produces offspring ``i``.

    Returns:
        Newly sampled individuals.
    """
    if one_each:
        parent_idxs = list(range(strategy.lamb))
    else:
        n_dom = _front(strategy.parents)
        parent_idxs = [n_dom[rng.integers(0, len(n_dom))].ps_[1] for _ in range(strategy.lamb)]
    raws = numpy.empty((strategy.lamb, strategy.dim), dtype=float)
    for i, p_idx in enumerate(parent_idxs):
        raws[i] = _raw(
            strategy.parents[p_idx], strategy.sigmas[p_idx], strategy.big_a[p_idx], arz[i]
        )
    if strategy.low is not None or strategy.up is not None:
        low_seq = broadcast_param(
            "low", -numpy.inf if strategy.low is None else strategy.low, strategy.dim
        )
        up_seq = broadcast_param(
            "up", numpy.inf if strategy.up is None else strategy.up, strategy.dim
        )
        raws = numpy.clip(
            raws, numpy.asarray(low_seq, dtype=float), numpy.asarray(up_seq, dtype=float)
        )
    individuals = []
    for i, p_idx in enumerate(parent_idxs):
        init = ind_init(raws[i])
        init.ps_ = "o", p_idx
        individuals.append(init)
    return individuals

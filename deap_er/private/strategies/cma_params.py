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
from math import log, sqrt
from typing import Any

import numpy

from .common import sample_offspring

__all__ = [
    "CmaCore",
    "adapt_cma_sigma",
    "apply_cma_hyperparams",
    "generate_cma_offspring",
    "init_cma_state",
    "reset_cma_state",
    "shift_cma_centroid",
    "update_cma_paths",
]


class CmaCore:
    """Typed mutable state shared by full-matrix and separable CMA."""

    update_count: int
    centroid: numpy.ndarray
    sigma: float
    dim: int
    pc: numpy.ndarray
    ps: numpy.ndarray
    chi_n: float
    lamb: int
    mu: int
    weights: numpy.ndarray
    mu_eff: float
    rank_one: float
    rank_mu: float
    ss_cum: float
    ss_dmp: float
    cm_cum: float
    big_c: numpy.ndarray
    diag_d: numpy.ndarray
    cond: float
    low: Any
    up: Any
    bound_mode: str
    resample_limit: int


def init_cma_state(strategy: Any, centroid: Any, sigma: float) -> None:
    """Initialize centroid, paths, and ``chi_n`` on a CMA strategy.

    Args:
        strategy: Strategy instance to mutate.
        centroid: Starting search point.
        sigma: Initial step size.
    """
    strategy.update_count = 0
    strategy.centroid = numpy.array(centroid)
    strategy.sigma = sigma
    strategy.dim = len(strategy.centroid)
    strategy.pc = numpy.zeros(strategy.dim)
    strategy.ps = numpy.zeros(strategy.dim)
    temp = 1 - 1.0 / (4.0 * strategy.dim) + 1.0 / (21.0 * strategy.dim**2)
    strategy.chi_n = sqrt(strategy.dim) * temp


def reset_cma_state(strategy: Any, centroid: Any, sigma: float) -> None:
    """Clear paths and step size for a CMA restart.

    Args:
        strategy: Strategy instance to mutate.
        centroid: Restart search point.
        sigma: Restart step size.
    """
    strategy.centroid = numpy.asarray(centroid, dtype=float)
    strategy.sigma = sigma
    strategy.pc = numpy.zeros(strategy.dim)
    strategy.ps = numpy.zeros(strategy.dim)
    strategy.update_count = 0


def generate_cma_offspring(
    strategy: Any, transform: numpy.ndarray, ind_init: Callable[..., Any]
) -> list[Any]:
    """Sample ``strategy.lamb`` individuals through ``transform``.

    Args:
        strategy: CMA strategy with centroid, sigma, bounds, and ``lamb``.
        transform: Full eigenbasis product or a length-``n`` scale vector.
        ind_init: Callable that turns a sampled vector into an individual.

    Returns:
        Newly sampled individuals.
    """
    return sample_offspring(
        strategy.centroid,
        strategy.sigma,
        transform,
        strategy.lamb,
        strategy.dim,
        ind_init,
        low=strategy.low,
        up=strategy.up,
        bound_mode=strategy.bound_mode,
        resample_limit=strategy.resample_limit,
    )


def shift_cma_centroid(strategy: Any, population: list[Any]) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Rank ``population`` and move the centroid to the weighted mean.

    Args:
        strategy: CMA strategy with ``weights`` and ``mu``.
        population: Evaluated individuals from ``generate``.

    Returns:
        The previous centroid and the centroid displacement.
    """
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    old_centroid = strategy.centroid
    strategy.centroid = numpy.dot(strategy.weights, numpy.asarray(population[0 : strategy.mu]))
    return old_centroid, strategy.centroid - old_centroid


def update_cma_paths(strategy: Any, c_diff: numpy.ndarray, y_mean: numpy.ndarray) -> float:
    """Update ``ps`` and ``pc``; return the stall gate ``hsig``.

    Args:
        strategy: CMA strategy with cumulation constants.
        c_diff: Centroid displacement.
        y_mean: Displacement in the whitened coordinate system.

    Returns:
        ``1.0`` when the rank-one path should advance, else ``0.0``.
    """
    temp_1 = sqrt(strategy.ss_cum * (2 - strategy.ss_cum) * strategy.mu_eff)
    strategy.ps = (1 - strategy.ss_cum) * strategy.ps + temp_1 / strategy.sigma * y_mean
    temp_1 = sqrt(1.0 - (1.0 - strategy.ss_cum) ** (2.0 * (strategy.update_count + 1.0)))
    stalled = numpy.linalg.norm(strategy.ps) / temp_1 / strategy.chi_n
    hsig = float(stalled < (1.4 + 2.0 / (strategy.dim + 1.0)))
    temp_1 = sqrt(strategy.cm_cum * (2 - strategy.cm_cum) * strategy.mu_eff)
    strategy.pc = (1 - strategy.cm_cum) * strategy.pc + hsig * temp_1 / strategy.sigma * c_diff
    return hsig


def adapt_cma_sigma(strategy: Any) -> None:
    """Apply cumulative step-size adaptation to ``strategy.sigma``.

    Args:
        strategy: CMA strategy with ``ps``, ``chi_n``, and damping.
    """
    temp = numpy.linalg.norm(strategy.ps) / strategy.chi_n - 1.0
    strategy.sigma *= numpy.exp(temp * strategy.ss_cum / strategy.ss_dmp)


def apply_cma_hyperparams(
    strategy: Any, kwargs: dict[str, Any], *, rank_scale: float = 1.0
) -> None:
    """Set λ, μ, weights, and CMA learning rates on ``strategy``.

    Default ``rank_one`` and ``rank_mu`` are multiplied by ``rank_scale``
    when those keys are omitted. Explicit kwargs are left unscaled.

    Args:
        strategy: CMA strategy with a ``dim`` attribute.
        kwargs: Constructor or ``compute_params`` keyword arguments.
        rank_scale: Multiplier for default rank-one and rank-μ rates.

    Raises:
        RuntimeError: If ``weights`` is not ``superlinear``,
            ``linear``, or ``equal``.
    """
    dim = strategy.dim
    default = int(4 + 3 * log(dim))
    strategy.lamb = int(kwargs.get("offsprings", default))
    default_mu = int(strategy.lamb / 2)
    if strategy.lamb >= 1:
        default_mu = max(1, default_mu)
    strategy.mu = int(kwargs.get("survivors", default_mu))
    scheme = kwargs.get("weights", "superlinear")
    if scheme == "superlinear":
        weights = log(strategy.mu + 0.5) - numpy.log(numpy.arange(1, strategy.mu + 1))
    elif scheme == "linear":
        weights = strategy.mu + 0.5 - numpy.arange(1, strategy.mu + 1)
    elif scheme == "equal":
        weights = numpy.ones(strategy.mu)
    else:
        raise RuntimeError(f"Unknown weights : {scheme}")
    strategy.weights = numpy.asarray(weights, dtype=float)
    strategy.weights /= sum(strategy.weights)
    strategy.mu_eff = 1.0 / sum(strategy.weights**2)
    default_one = 2.0 / ((dim + 1.3) ** 2 + strategy.mu_eff)
    if "rank_one" in kwargs:
        strategy.rank_one = float(kwargs["rank_one"])
    else:
        strategy.rank_one = float(default_one * rank_scale)
    temp_1 = strategy.mu_eff - 2.0 + 1.0 / strategy.mu_eff
    default_mu = 2.0 * temp_1 / ((dim + 2.0) ** 2 + strategy.mu_eff)
    if "rank_mu" in kwargs:
        strategy.rank_mu = float(kwargs["rank_mu"])
    else:
        strategy.rank_mu = float(default_mu * rank_scale)
    strategy.rank_mu = min(1 - strategy.rank_one, strategy.rank_mu)
    default = (strategy.mu_eff + 2.0) / (dim + strategy.mu_eff + 3.0)
    strategy.ss_cum = float(kwargs.get("ss_cum", default))
    temp_1 = sqrt((strategy.mu_eff - 1.0) / (dim + 1.0))
    default = 1.0 + 2.0 * max(0.0, temp_1 - 1.0) + strategy.ss_cum
    strategy.ss_dmp = float(kwargs.get("ss_dmp", default))
    strategy.cm_cum = float(kwargs.get("cm_cum", 4.0 / (dim + 4.0)))

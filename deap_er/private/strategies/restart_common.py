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
from math import ceil, log, sqrt
from typing import TYPE_CHECKING, Any

import numpy

from deap_er.private.strategies.common import finite_sample_bounds
from deap_er.private.various.rng import rng

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = [
    "default_lambda",
    "max_iter_limit",
    "require_stagnation_key",
    "sample_centroid",
    "sample_small_lambda",
    "sample_small_sigma",
    "scalar_fitness",
    "stagnation_window_size",
    "strategy_center",
    "strategy_sigma",
    "strategy_diagnostics",
    "strategy_dim",
]


def default_lambda(dim: int) -> int:
    """Return the default CMA offspring count for dimension ``dim``."""
    return int(4 + 3 * log(dim))


def max_iter_limit(dim: int, lamb: int, base: int = 1000, scale: float = 500.0) -> int:
    """Return the per-run generation cap used by BBOB-style restarts."""
    return int(base + scale * (dim + 3) ** 2 / sqrt(lamb))


def stagnation_window_size(gen: int, dim: int, lamb: int) -> int:
    """Return the stagnation history length for generation ``gen``."""
    return int(ceil(0.2 * gen + 120 + 30 * dim / lamb))


def sample_centroid(
    dim: int,
    low: Any,
    up: Any,
    mode: str | Callable[[int], numpy.ndarray],
    initial: numpy.ndarray,
    best: Individual | None,
) -> numpy.ndarray:
    """Draw a restart centroid according to ``mode``.

    Args:
        dim: Search-space dimension.
        low: Optional lower box bound.
        up: Optional upper box bound.
        mode: ``random``, ``initial``, ``best``, or a callable.
        initial: Centroid from the first run.
        best: Best individual so far, if any.

    Returns:
        A length-``dim`` centroid vector.
    """
    if callable(mode):
        return numpy.asarray(mode(dim), dtype=float)  # ty: ignore[call-top-callable]
    if mode == "initial":
        return numpy.asarray(initial, dtype=float)
    if mode == "best" and best is not None:
        return numpy.asarray(best, dtype=float)
    low_arr, up_arr = finite_sample_bounds(low, up, dim)
    return numpy.array([rng.uniform(lo, hi) for lo, hi in zip(low_arr, up_arr, strict=True)])


def sample_small_lambda(lambda_default: int, lambda_large: int) -> int:
    """Sample a BIPOP small-regime offspring count.

    Hansen (2009): ``λ_s = ⌊λ_def · (λ_large / (2 λ_def))^{U²}⌋`` with
    ``U ~ U[0, 1]``. Squaring ``U`` biases draws toward ``λ_def``.

    Args:
        lambda_default: Default (first-run) offspring count.
        lambda_large: Offspring count of the latest large-regime run.

    Returns:
        An offspring count of at least 1.
    """
    ratio = lambda_large / (2.0 * lambda_default)
    return max(1, int(lambda_default * ratio ** (rng.random() ** 2)))


def sample_small_sigma(sigma_large: float = 2.0) -> float:
    """Sample a BIPOP small-regime step size ``sigma_large * 10^{-2U}``.

    Args:
        sigma_large: First-run / large-regime step size (Hansen's ``σ0``).

    Returns:
        A step size in ``[0.01 * sigma_large, sigma_large]``.
    """
    return float(sigma_large * 10.0 ** (-2.0 * rng.random()))


def scalar_fitness(ind: Individual, key: Callable[[Individual], float] | None = None) -> float:
    """Return a weighted scalar for stagnation checks (higher is better).

    For single-objective fitness, uses ``wvalues[0]``. For multi-objective
    fitness, ``key`` must be supplied — there is no default scalarization.

    Args:
        ind: Evaluated individual.
        key: Optional callable returning a higher-is-better stagnation scalar.

    Raises:
        ValueError: If ``key`` is omitted for multi-objective fitness.
    """
    if key is not None:
        return float(key(ind))
    wvalues = ind.fitness.wvalues
    if len(wvalues) == 1:
        return float(wvalues[0])
    raise ValueError(
        "multi-objective stagnation requires an explicit stagnation_key "
        "callable returning a higher-is-better scalar"
    )


def require_stagnation_key(
    weights: tuple[float, ...] | None,
    stagnation_key: Callable[[Individual], float] | None,
) -> None:
    """Raise if multi-objective fitness has no ``stagnation_key``."""
    if weights is not None and len(weights) > 1 and stagnation_key is None:
        raise ValueError(
            "multi-objective stagnation requires an explicit stagnation_key "
            "callable returning a higher-is-better scalar"
        )


def strategy_dim(strategy: Any) -> int:
    """Return the search-space dimension of a CMA strategy."""
    return int(strategy.dim)


def strategy_center(strategy: Any) -> numpy.ndarray:
    """Return the initial search center of a CMA strategy."""
    if hasattr(strategy, "centroid"):
        return numpy.asarray(strategy.centroid, dtype=float)
    if hasattr(strategy, "parent"):
        return numpy.asarray(strategy.parent, dtype=float)
    return numpy.asarray(strategy.parents[0], dtype=float)


def strategy_sigma(strategy: Any) -> float:
    """Return the current step size of a CMA strategy."""
    if hasattr(strategy, "sigmas"):
        return float(strategy.sigmas[0])
    return float(strategy.sigma)


def strategy_diagnostics(strategy: Any) -> tuple[float | None, float | None, float | None]:
    """Return optional condition number, sigma, and largest covariance eigenvalue."""
    if hasattr(strategy, "cond"):
        largest = float(numpy.max(strategy.diag_d) ** 2) if len(strategy.diag_d) else 1.0
        return float(strategy.cond), strategy_sigma(strategy), largest
    return None, strategy_sigma(strategy), None

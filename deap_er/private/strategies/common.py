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
from math import exp
from typing import TYPE_CHECKING, Any

import numpy

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual, NumOrSeq
from deap_er.private.operators.bounds import broadcast_param
from deap_er.private.various.rng import rng

__all__: list[str] = [
    "sample_offspring",
    "step_size_multiplier",
    "update_bound_attrs",
    "apply_box_bounds",
    "finite_sample_bounds",
]


def update_bound_attrs(strategy: Any, kwargs: dict[str, Any]) -> None:
    """Store box-bound kwargs on ``strategy`` without clearing omitted keys.

    Args:
        strategy: CMA strategy whose bound attributes to update.
        kwargs: Constructor or ``compute_params`` keyword arguments.

    Raises:
        ValueError: If ``bound_mode`` is not ``clip`` or ``resample``,
            or if ``resample_limit`` is less than 1.
    """
    if "low" in kwargs:
        strategy.low = kwargs["low"]
    elif not hasattr(strategy, "low"):
        strategy.low = None
    if "up" in kwargs:
        strategy.up = kwargs["up"]
    elif not hasattr(strategy, "up"):
        strategy.up = None
    if "bound_mode" in kwargs:
        strategy.bound_mode = kwargs["bound_mode"]
    elif not hasattr(strategy, "bound_mode"):
        strategy.bound_mode = "clip"
    if "resample_limit" in kwargs:
        strategy.resample_limit = int(kwargs["resample_limit"])
    elif not hasattr(strategy, "resample_limit"):
        strategy.resample_limit = 100
    if strategy.bound_mode not in ("clip", "resample"):
        raise ValueError("bound_mode must be 'clip' or 'resample'.")
    if strategy.resample_limit < 1:
        raise ValueError("resample_limit must be at least 1.")


def _bound_arrays(
    low: NumOrSeq | None, up: NumOrSeq | None, dim: int
) -> tuple[numpy.ndarray, numpy.ndarray] | None:
    """Broadcast optional box bounds to length ``dim``.

    Args:
        low: Lower bound, or None for unbounded below.
        up: Upper bound, or None for unbounded above.
        dim: Search-space dimension.

    Returns:
        ``(low, up)`` arrays, or None when both bounds are omitted.
    """
    if low is None and up is None:
        return None
    low_seq = broadcast_param("low", -numpy.inf if low is None else low, dim)
    up_seq = broadcast_param("up", numpy.inf if up is None else up, dim)
    return numpy.asarray(low_seq, dtype=float), numpy.asarray(up_seq, dtype=float)


def finite_sample_bounds(
    low: NumOrSeq | None, up: NumOrSeq | None, dim: int
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return a finite box for uniform centroid draws.

    Clip and resample still use infinite ends for a missing bound.
    A restart centroid cannot: ``uniform(0, ∞)`` is ``inf`` and
    ``uniform(-∞, 1)`` is ``NaN``. Missing or non-finite ends fall
    back to ``[-5, 5]``. If that collapses an axis, the open side
    grows by 10 (the default box width).

    Args:
        low: Lower bound, or None for unbounded below.
        up: Upper bound, or None for unbounded above.
        dim: Search-space dimension.

    Returns:
        Finite ``(low, up)`` arrays of length ``dim``.
    """
    default_lo, default_hi = -5.0, 5.0
    width = default_hi - default_lo
    bounds = _bound_arrays(low, up, dim)
    if bounds is None:
        return (
            numpy.full(dim, default_lo, dtype=float),
            numpy.full(dim, default_hi, dtype=float),
        )
    raw_lo, raw_hi = bounds
    lo = numpy.where(numpy.isfinite(raw_lo), raw_lo, default_lo)
    hi = numpy.where(numpy.isfinite(raw_hi), raw_hi, default_hi)
    missing_hi = ~numpy.isfinite(raw_hi)
    missing_lo = ~numpy.isfinite(raw_lo)
    hi = numpy.where(missing_hi & (lo >= hi), lo + width, hi)
    lo = numpy.where(missing_lo & (lo >= hi), hi - width, lo)
    return lo.astype(float), hi.astype(float)


def apply_box_bounds(
    vector: numpy.ndarray,
    low: NumOrSeq | None,
    up: NumOrSeq | None,
    bound_mode: str,
    resample_limit: int,
    redraw: Callable[[], numpy.ndarray],
) -> numpy.ndarray:
    """Clip or resample ``vector`` into the optional box.

    Both modes are constraint-handling approximations. A later CMA
    update treats the repaired point as the sample.

    Args:
        vector: Candidate sample.
        low: Lower bound, or None.
        up: Upper bound, or None.
        bound_mode: ``clip`` or ``resample``.
        resample_limit: Failed redraws before clipping the last draw.
        redraw: Draws a replacement vector for ``resample``.

    Returns:
        A vector inside the box, or ``vector`` when no bounds are set.
    """
    bounds = _bound_arrays(low, up, len(vector))
    if bounds is None:
        return vector
    low_arr, up_arr = bounds
    if bound_mode == "clip":
        return numpy.clip(vector, low_arr, up_arr)
    cand = numpy.asarray(vector, dtype=float)
    for _ in range(resample_limit):
        if numpy.all(cand >= low_arr) and numpy.all(cand <= up_arr):
            return cand
        cand = numpy.asarray(redraw(), dtype=float)
    return numpy.clip(cand, low_arr, up_arr)


def sample_offspring(
    center: numpy.ndarray | Individual,
    sigma: float,
    transform: numpy.ndarray,
    lamb: int,
    dim: int,
    ind_init: Callable[..., Individual],
    low: NumOrSeq | None = None,
    up: NumOrSeq | None = None,
    bound_mode: str = "clip",
    resample_limit: int = 100,
) -> list[Individual]:
    """Sample individuals from a multivariate normal distribution.

    Draws ``lamb`` standard normal vectors and maps them through
    ``transform`` scaled by ``sigma``, around ``center``. Optional
    box bounds clip or resample each vector before ``ind_init``.

    Args:
        center: Mean of the search distribution.
        sigma: Standard deviation of the distribution.
        transform: Matrix that shapes the distribution, usually a
            Cholesky factor or a scaled eigenbasis.
        lamb: Number of individuals to sample.
        dim: Dimensionality of the search space.
        ind_init: Callable that turns a sampled vector into an
            individual.
        low: Lower box bound. Optional.
        up: Upper box bound. Optional.
        bound_mode: ``clip`` or ``resample`` when bounds are set.
        resample_limit: Failed redraws before clipping one sample.

    Returns:
        Newly sampled individuals.
    """

    def _map_z(z: numpy.ndarray) -> numpy.ndarray:
        return numpy.asarray(center + sigma * numpy.dot(z, transform.T), dtype=float)

    bounds = _bound_arrays(low, up, dim)
    if bounds is None or bound_mode == "clip":
        arz = rng.standard_normal((lamb, dim))
        arz = _map_z(arz)
        if bounds is not None:
            arz = numpy.clip(arz, bounds[0], bounds[1])
        return list(map(ind_init, arz))

    individuals = []
    for _ in range(lamb):
        raw = _map_z(rng.standard_normal(dim))
        vector = apply_box_bounds(
            raw, low, up, "resample", resample_limit, lambda: _map_z(rng.standard_normal(dim))
        )
        individuals.append(ind_init(vector))
    return individuals


def step_size_multiplier(psucc: float, tgt_sr: float, ss_dmp: float) -> float:
    """Return the step-size factor implied by a success rate.

    The step-size grows while the success rate is above ``tgt_sr`` and
    shrinks below it.

    Args:
        psucc: Smoothed success rate.
        tgt_sr: Target success rate.
        ss_dmp: Damping of the step-size.

    Returns:
        The multiplier to apply to the current step-size.
    """
    return exp((psucc - tgt_sr) / (ss_dmp * (1.0 - tgt_sr)))

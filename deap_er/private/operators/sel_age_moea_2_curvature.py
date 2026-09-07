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

import numpy
from numpy import ndarray

__all__: list[str] = [
    "estimate_curvature_nr",
]

_ZERO_TOL = 1e-12
_NR_EPS = 1e-10
_LOG_MAX = numpy.log(numpy.finfo(float).max)


def _positive_power_sum(point: ndarray, p: float, epsilon: float) -> float | None:
    total = 0.0
    for value in point:
        if value <= 0.0:
            continue
        log_value = p * numpy.log(value + epsilon)
        if log_value >= _LOG_MAX:
            return None
        total += float(numpy.exp(log_value))
    return float(total)


def _nr_power_moments(point: ndarray, p: float, epsilon: float) -> tuple[float, float] | None:
    numerator = 0.0
    denominator = 0.0
    for value in point:
        if value <= 0.0:
            continue
        log_value = p * numpy.log(value + epsilon)
        if log_value >= _LOG_MAX:
            return None
        power = float(numpy.exp(log_value))
        log_term = float(numpy.log(value + epsilon))
        numerator += power * log_term
        denominator += power
    return float(numerator), float(denominator)


def _nr_next_p(point: ndarray, p: float) -> float | None:
    total = _positive_power_sum(point, p, _NR_EPS)
    if total is None:
        return None
    moments = _nr_power_moments(point, p, _NR_EPS)
    if moments is None:
        return None
    numerator, denominator = moments
    if abs(denominator) < _ZERO_TOL or not numpy.isfinite(numerator + denominator):
        return None
    deriv = numerator / denominator
    if abs(deriv) < _ZERO_TOL:
        return None
    func = numpy.log(total) if total > 0.0 else 0.0
    return float(p - func / deriv)


def estimate_curvature_nr(
    point: ndarray,
    n_obj: int,
    tol: float = 1e-3,
    max_iter: int = 100,
) -> float:
    """Estimate the Pareto-front curvature ``p`` with Newton-Raphson.

    Solves ``sum(x_i^p) = 1`` for a normalized non-dominated point.

    Args:
        point: Normalized objective vector with shape ``(m,)``.
        n_obj: Number of objectives.
        tol: Stop when successive ``p`` estimates differ by at most this value.
        max_iter: Maximum Newton-Raphson iterations.

    Returns:
        Estimated curvature ``p`` in ``[0.1, 20]``, or ``1.0`` on failure.
    """
    if point.shape[0] != n_obj:
        return 1.0

    p = 1.0
    past = p

    for _ in range(max_iter):
        next_p = _nr_next_p(point, p)
        if next_p is None:
            return 1.0
        p = next_p
        if abs(p - past) <= tol:
            break
        past = p

    if not numpy.isfinite(p) or p <= 0.1:
        return 1.0
    if p > 20.0:
        return 20.0
    return float(p)

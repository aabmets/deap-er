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

from deap_er.private.various.case_errors import case_valid_mask

__all__: list[str] = ["affine_scale"]


def _as_series(
    predicted: numpy.ndarray,
    target: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    left = numpy.asarray(predicted, dtype=numpy.float64)
    right = numpy.asarray(target, dtype=numpy.float64)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("predicted and target must be one-dimensional arrays")
    if left.shape[0] != right.shape[0]:
        raise ValueError("predicted and target must have the same length")
    return left, right


def affine_scale(
    predicted: numpy.ndarray,
    target: numpy.ndarray,
    *,
    valid: numpy.ndarray | None = None,
) -> tuple[float, float]:
    """Fit Keijzer intercept and slope for ``a + b * f(x)``.

    Least-squares ``a`` and ``b`` are computed on the same scorable
    samples :func:`~deap_er.tools.case_errors` uses: an optional
    ``valid=`` mask intersected with the finite check on both series.
    Darwinian callers apply ``a + b * predicted`` only when writing
    fitness or case errors; the tree is unchanged.

    A series with no scorable samples returns the identity
    ``(0.0, 1.0)``. A constant prediction returns an intercept-only
    shift ``(mean(target) - mean(predicted), 1.0)``.

    Args:
        predicted: Predicted series ``f(x)``.
        target: Target series, same length as ``predicted``.
        valid: Optional per-sample mask. Same contract as
            :func:`~deap_er.tools.case_errors`.

    Returns:
        ``(a, b)`` so the scaled series is ``a + b * predicted``.

    Raises:
        ValueError: If the inputs are not aligned one-dimensional
            arrays, or if ``valid`` has the wrong shape.
    """
    predicted, target = _as_series(predicted, target)
    sample_valid = case_valid_mask(predicted, target, valid)
    if not numpy.any(sample_valid):
        return 0.0, 1.0
    forecast = predicted[sample_valid]
    observed = target[sample_valid]
    forecast_mean = float(numpy.mean(forecast))
    observed_mean = float(numpy.mean(observed))
    centered = forecast - forecast_mean
    denom = float(numpy.dot(centered, centered))
    if not numpy.isfinite(denom) or denom <= 0.0:
        return observed_mean - forecast_mean, 1.0
    slope = float(numpy.dot(observed - observed_mean, centered) / denom)
    intercept = observed_mean - slope * forecast_mean
    return intercept, slope

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

from collections.abc import Sequence

import numpy

from deap_er.private.various.affine_scale import affine_scale
from deap_er.private.various.case_errors import case_errors

__all__: list[str] = ["affine_case_errors"]


def affine_case_errors(
    predicted: numpy.ndarray,
    target: numpy.ndarray,
    ranges: Sequence[tuple[int, int]] | numpy.ndarray,
    *,
    valid: numpy.ndarray | None = None,
    empty: float = float("inf"),
) -> tuple[float, ...]:
    r"""Return Keijzer-scaled case errors without changing the tree.

    Fits $a + b\,f(x)$ with :func:`~deap_er.tools.affine_scale` on the
    same ``valid=`` mask :func:`~deap_er.tools.case_errors` uses, applies
    the scaled series, then reduces to one MSE per case. This is the
    Darwinian default next to Lamarckian
    :func:`~deap_er.gp.write_affine_scale`.

    Args:
        predicted: Predicted series ``f(x)``.
        target: Target series, same length as ``predicted``.
        ranges: Case bounds forwarded to :func:`~deap_er.tools.case_errors`.
        valid: Optional per-sample mask forwarded to both helpers.
        empty: Value returned when a case has no scorable samples.

    Returns:
        One MSE per case, in range order.

    Raises:
        ValueError: If the inputs are not aligned one-dimensional
            arrays or if ``valid`` has the wrong shape.
    """
    intercept, slope = affine_scale(predicted, target, valid=valid)
    scaled = intercept + slope * numpy.asarray(predicted, dtype=numpy.float64)
    return case_errors(scaled, target, ranges, valid=valid, empty=empty)

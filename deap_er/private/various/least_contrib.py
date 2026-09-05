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

from typing import TYPE_CHECKING

import moocore
import numpy

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

from .hypervolume import minimized_points

__all__: list[str] = ["least_contrib"]


def least_contrib(
    population: list[Individual], ref_point: list[float] | numpy.ndarray | None = None
) -> int:
    """Return the index of the individual with the least hypervolume contribution.

    Minimization is implicitly assumed. ``ref_point`` is interpreted in
    that same space (after ``wvalues`` are negated). Delegates to
    ``moocore.hv_contributions``.

    Args:
        population: Non-dominated individuals, each with a Fitness
            attribute.
        ref_point: Reference point for the hypervolume. Optional. If
            omitted, the worst value of each objective plus one is
            used.

    Returns:
        The index of the individual with the least hypervolume
        contribution. The first index wins when contributions tie.

    Raises:
        ValueError: If ``population`` is empty.
    """
    if not population:
        raise ValueError("population must not be empty")
    wvals = minimized_points(population)
    point = numpy.max(wvals, axis=0) + 1 if ref_point is None else numpy.asarray(ref_point)
    contrib = moocore.hv_contributions(wvals, ref=point, maximise=False)
    return int(numpy.argmin(contrib))

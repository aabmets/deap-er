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

from itertools import chain
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual
from deap_er.private.various.hypervolume import minimized_points
from deap_er.private.various.least_contrib import least_contrib
from deap_er.private.various.sort_non_dominated import sort_non_dominated

__all__: list[str] = ["sel_sms_emoa"]


def sel_sms_emoa(
    individuals: list[Individual],
    sel_count: int,
    ref_point: list[float] | numpy.ndarray | None = None,
) -> list[Individual]:
    """Select the next generation with SMS-EMOA.

    Non-dominated sorting ranks the pool. Complete fronts are kept.
    When the next front would exceed ``sel_count``, individuals with
    the smallest hypervolume contribution on that critical front are
    removed one at a time until the quota is met. This is the Reduce
    operator from Beume, Naujoks, and Emmerich (2007).

    Use on ``parents + offspring`` for generational search, or on
    ``parents + [child]`` for steady-state ``(mu + 1)`` selection.

    Args:
        individuals: Evaluated individuals to select from.
        sel_count: Number of individuals to keep.
        ref_point: Reference point in minimization space (the same
            convention as ``hypervolume`` and ``least_contrib``:
            internally ``-wvalues``). Optional. When omitted, the
            worst objective value in the input pool plus one is used
            and kept fixed for the whole truncation pass.

    Returns:
        The selected individuals, ordered by Pareto front.
    """
    if not individuals or sel_count <= 0:
        return []
    if sel_count >= len(individuals):
        return list(chain(*sort_non_dominated(individuals, len(individuals))))

    pareto_fronts = sort_non_dominated(individuals, sel_count)

    chosen = list(chain(*pareto_fronts[:-1]))
    need = sel_count - len(chosen)
    critical = pareto_fronts[-1]

    if need < len(critical):
        ref = (
            numpy.asarray(ref_point)
            if ref_point is not None
            else numpy.max(minimized_points(individuals), axis=0) + 1
        )
        while len(critical) > need:
            idx = least_contrib(critical, ref)
            critical.pop(idx)

    chosen.extend(critical)
    return chosen

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

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__: list[str] = ["mig_ring"]


def mig_ring(
    populations: list[list[Individual]],
    mig_count: int,
    selection: Callable[..., Any],
    replacement: Callable[..., Any] | None = None,
    mig_indices: list[int] | None = None,
) -> None:
    """Move emigrants between populations along a ring (or custom map).

    From each population, ``selection`` picks ``mig_count`` emigrants.
    Those individuals replace members of the destination population.
    Populations are modified in place.

    Args:
        populations: Populations to migrate between.
        mig_count: Number of individuals to migrate from each population.
        selection: Callable that selects emigrants from a population.
        replacement: Callable that selects which destination individuals
            are replaced. If omitted, the destination's own emigrants
            are the vacancies.
        mig_indices: Destination index for each source population. If
            omitted, each population sends to the next and the last
            wraps to the first.
    """
    nbr_demes = len(populations)
    if mig_indices is None:
        mig_indices = list(range(1, nbr_demes)) + [0]

    immigrants = [[] for _ in range(nbr_demes)]
    emigrants = [[] for _ in range(nbr_demes)]

    for from_deme in range(nbr_demes):
        emigrants[from_deme].extend(selection(populations[from_deme], mig_count))
        if replacement is None:
            immigrants[from_deme] = emigrants[from_deme]
        else:
            immigrants[from_deme].extend(replacement(populations[from_deme], mig_count))

    for from_deme, to_deme in enumerate(mig_indices):
        for i, immigrant in enumerate(immigrants[to_deme]):
            indx = next(j for j, member in enumerate(populations[to_deme]) if member is immigrant)
            populations[to_deme][indx] = emigrants[from_deme][i]

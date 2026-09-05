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
from deap_er.base.typedefs import Individual

from ._spea_2 import _fill_from_density, _raw_fitness
from ._spea_2_archive import _truncate_archive

__all__ = ["sel_spea_2"]


def sel_spea_2(individuals: list[Individual], sel_count: int) -> list[Individual]:
    """Select the next generation with SPEA-II.

    The pool is usually larger than ``sel_count``. If the two sizes
    are equal, the population is sorted by Pareto front.

    Args:
        individuals: Individuals to select from.
        sel_count: Number of individuals to select.

    Returns:
        The selected individuals.
    """
    fits = _raw_fitness(individuals)

    chosen = [i for i in range(len(individuals)) if fits[i] < 1]
    if len(chosen) < sel_count:
        chosen = _fill_from_density(individuals, chosen, fits, sel_count)
    elif len(chosen) > sel_count:
        chosen = _truncate_archive(individuals, chosen, sel_count)

    return [individuals[i] for i in chosen]

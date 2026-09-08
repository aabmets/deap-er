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

import numpy

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual, Mutant, NumOrSeq
from deap_er.private.various.rng import rng

from .bounds import broadcast_param

__all__: list[str] = [
    "iso_line_bit",
    "iso_line_float",
    "iso_line_int",
    "mut_iso_line",
]

_DONOR_SHORT = "Donor must be at least the size of the individual"
_BOUNDS_PAIR = "Arguments 'low' and 'up' must both be set or both omitted."


def _is_bool_gene(value: object) -> bool:
    return type(value) is bool or isinstance(value, numpy.bool_)


def _sample_t(iso: float) -> float:
    return rng.uniform(-iso, 1.0 + iso)


def iso_line_float(
    parent: float,
    donor: float,
    iso: float,
    sigma: float,
    *,
    low: float | None = None,
    up: float | None = None,
) -> float:
    """Interpolate toward ``donor`` and add isotropic Gaussian noise.

    Args:
        parent: Current gene value.
        donor: Elite donor value.
        iso: Line extension for ``t ~ Uniform(-iso, 1 + iso)``.
        sigma: Standard deviation of the isotropic perturbation.
        low: Optional lower bound applied after the draw.
        up: Optional upper bound applied after the draw.

    Returns:
        The mutated gene value.
    """
    gene = parent + _sample_t(iso) * (donor - parent) + rng.gauss(0.0, sigma)
    if low is not None and up is not None and up > low:
        gene = min(max(gene, low), up)
    return float(gene)


def iso_line_int(
    parent: int,
    donor: int,
    iso: float,
    sigma: float,
    *,
    low: int,
    up: int,
) -> int:
    """Interpolate toward ``donor``, add noise, round, and clamp.

    Args:
        parent: Current gene value.
        donor: Elite donor value.
        iso: Line extension for ``t ~ Uniform(-iso, 1 + iso)``.
        sigma: Standard deviation of the isotropic perturbation.
        low: Inclusive lower bound.
        up: Inclusive upper bound.

    Returns:
        The mutated integer gene.
    """
    raw = parent + _sample_t(iso) * (donor - parent) + rng.gauss(0.0, sigma)
    if up < low:
        return int(parent)
    return int(min(max(int(round(raw)), low), up))


def iso_line_bit(parent: bool, donor: bool, iso: float, sigma: float) -> bool:
    """Move toward ``donor`` along the line, then apply isotropic flips.

    ``t`` maps to a Bernoulli draw toward ``donor``. ``sigma`` is the
    probability of flipping the chosen bit afterward.

    Args:
        parent: Current gene value.
        donor: Elite donor value.
        iso: Line extension for ``t ~ Uniform(-iso, 1 + iso)``.
        sigma: Flip probability after the line draw.

    Returns:
        The mutated boolean gene.
    """
    t = _sample_t(iso)
    value = bool(donor) if rng.random() < max(0.0, min(1.0, t)) else bool(parent)
    if sigma > 0.0 and rng.random() < min(1.0, sigma):
        value = not value
    return value


def mut_iso_line(
    individual: Individual,
    donor: Individual,
    iso: float,
    sigma: float,
    *,
    low: NumOrSeq | None = None,
    up: NumOrSeq | None = None,
) -> Mutant:
    """Apply iso+line mutation toward an archive elite donor.

    Each gene becomes ``parent + t * (donor - parent) + N(0, sigma)``
    with ``t ~ Uniform(-iso, 1 + iso)``. Booleans use ``iso_line_bit``;
    integers (that are not bool) round and clamp when bounds are given;
    other numeric genes clamp when bounds are given. The individual is
    modified in place.

    Args:
        individual: Parent to overwrite. Clone first when needed.
        donor: Elite donor from ``archive.random_elites``.
        iso: Line extension parameter.
        sigma: Isotropic noise standard deviation (flip rate for bool).
        low: Lower search bound. Optional.
        up: Upper search bound. Optional.

    Returns:
        A one-element tuple containing the mutated individual.

    Raises:
        ValueError: If ``donor`` is shorter than ``individual``, or if
            only one of ``low`` / ``up`` is set, or if a bound sequence
            is shorter than the individual.
    """
    size = len(individual)
    if len(donor) < size:
        raise ValueError(f"{_DONOR_SHORT}: {len(donor)} < {size}")
    if (low is None) ^ (up is None):
        raise ValueError(_BOUNDS_PAIR)

    lows = ups = None
    if low is not None and up is not None:
        lows = broadcast_param("low", low, size)
        ups = broadcast_param("up", up, size)

    for index in range(size):
        parent = individual[index]
        elite = donor[index]
        if _is_bool_gene(parent):
            value = iso_line_bit(bool(parent), bool(elite), iso, sigma)
            individual[index] = type(parent)(value)
            continue
        if isinstance(parent, int):
            xl = int(lows[index]) if lows is not None else int(parent)
            xu = int(ups[index]) if ups is not None else int(parent)
            if lows is None or ups is None or xu < xl:
                gene = parent + _sample_t(iso) * (elite - parent) + rng.gauss(0.0, sigma)
                individual[index] = int(round(gene))
            else:
                individual[index] = iso_line_int(parent, int(elite), iso, sigma, low=xl, up=xu)
            continue
        xl = lows[index] if lows is not None else None
        xu = ups[index] if ups is not None else None
        individual[index] = iso_line_float(parent, elite, iso, sigma, low=xl, up=xu)

    return (individual,)

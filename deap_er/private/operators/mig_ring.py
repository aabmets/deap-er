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
from deap_er.private.various.clone import clone_individual

__all__: list[str] = ["mig_ring"]


def _claim_vacancies(population: list[Individual], dest_slots: list[Individual]) -> list[int]:
    taken: set[int] = set()
    vacancies: list[int] = []
    for immigrant in dest_slots:
        indx = next(
            (j for j, member in enumerate(population) if member is immigrant and j not in taken),
            None,
        )
        if indx is None:
            indx = next(
                (j for j in range(len(population)) if j not in taken),
                None,
            )
            if indx is None:
                break
        taken.add(indx)
        vacancies.append(indx)
    return vacancies


def _incoming_filled(
    emigrants: list[list[Individual]],
    vacancies: list[list[int]],
    mig_indices: list[int],
) -> list[int]:
    incoming = [0] * len(emigrants)
    for from_deme, to_deme in enumerate(mig_indices):
        filled = min(len(emigrants[from_deme]), len(vacancies[to_deme]))
        incoming[to_deme] = max(incoming[to_deme], filled)
    return incoming


def _place_emigrants(
    populations: list[list[Individual]],
    emigrants: list[list[Individual]],
    vacancies: list[list[int]],
    mig_indices: list[int],
    incoming_filled: list[int],
    replacement: Callable[..., Any] | None,
) -> None:
    for from_deme, to_deme in enumerate(mig_indices):
        for offset, (indx, immigrant) in enumerate(
            zip(vacancies[to_deme], emigrants[from_deme], strict=False)
        ):
            mover = immigrant
            if replacement is None and offset >= incoming_filled[from_deme]:
                mover = clone_individual(immigrant)
            populations[to_deme][indx] = mover


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
    When a source sends more emigrants than the destination has
    vacancies, or a deme is smaller than ``mig_count``, only as
    many individuals as both sides can hold are moved. Deme
    lengths are unchanged. When ``replacement`` is omitted, an
    emigrant whose home vacancy is not filled is cloned so the
    same object is not left in two demes. Populations are
    modified in place.

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

    emigrants: list[list[Individual]] = [[] for _ in range(nbr_demes)]
    vacancies: list[list[int]] = [[] for _ in range(nbr_demes)]

    for from_deme in range(nbr_demes):
        selected = selection(populations[from_deme], mig_count)
        if replacement is None:
            emigrants[from_deme].extend(selected)
            dest_slots = selected
        else:
            emigrants[from_deme].extend(clone_individual(ind) for ind in selected)
            dest_slots = replacement(populations[from_deme], mig_count)
        vacancies[from_deme] = _claim_vacancies(populations[from_deme], dest_slots)

    incoming_filled = _incoming_filled(emigrants, vacancies, mig_indices)
    _place_emigrants(
        populations,
        emigrants,
        vacancies,
        mig_indices,
        incoming_filled,
        replacement,
    )

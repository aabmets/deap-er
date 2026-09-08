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
from deap_er.private.various.rng import rng

__all__: list[str] = ["mig_fully_connected", "mig_random", "mig_ring"]


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
        dest = populations[to_deme]
        for offset, (indx, immigrant) in enumerate(
            zip(vacancies[to_deme], emigrants[from_deme], strict=False)
        ):
            already_in_dest = any(member is immigrant for member in dest)
            if (replacement is None and offset >= incoming_filled[from_deme]) or already_in_dest:
                mover = clone_individual(immigrant)
            else:
                mover = immigrant
            dest[indx] = mover


def _mig_edge(
    populations: list[list[Individual]],
    from_deme: int,
    to_deme: int,
    mig_count: int,
    selection: Callable[..., Any],
    replacement: Callable[..., Any] | None,
) -> None:
    src = populations[from_deme]
    dest = populations[to_deme]
    selected = selection(src, mig_count)
    if replacement is None:
        emigrants = selected
        dest_slots = selection(dest, mig_count)
    else:
        emigrants = [clone_individual(ind) for ind in selected]
        dest_slots = replacement(dest, mig_count)
    vacancies = _claim_vacancies(dest, dest_slots)
    incoming_filled = min(len(emigrants), len(vacancies))
    for offset, (indx, immigrant) in enumerate(zip(vacancies, emigrants, strict=False)):
        already_in_dest = any(member is immigrant for member in dest)
        if (replacement is None and offset >= incoming_filled) or already_in_dest:
            mover = clone_individual(immigrant)
        else:
            mover = immigrant
        dest[indx] = mover


def mig_fully_connected(
    populations: list[list[Individual]],
    mig_count: int,
    selection: Callable[..., Any],
    replacement: Callable[..., Any] | None = None,
) -> None:
    """Move emigrants along every directed island edge.

    For each ordered pair of distinct demes ``(src, dst)``, ``selection``
    picks ``mig_count`` emigrants from ``src`` and writes them into
    ``dst`` using the same vacancy and cloning rules as ``mig_ring``.
    Deme lengths are unchanged. Populations are modified in place.

    Args:
        populations: Populations to migrate between.
        mig_count: Number of individuals to migrate along each edge.
        selection: Callable that selects emigrants from a population.
        replacement: Callable that selects destination vacancies in the
            receiving deme. If omitted, the receiver's own emigrant
            slots are the vacancies.
    """
    nbr_demes = len(populations)
    for to_deme in range(nbr_demes):
        for from_deme in range(nbr_demes):
            if from_deme != to_deme:
                _mig_edge(
                    populations,
                    from_deme,
                    to_deme,
                    mig_count,
                    selection,
                    replacement,
                )


def mig_random(
    populations: list[list[Individual]],
    mig_count: int,
    selection: Callable[..., Any],
    replacement: Callable[..., Any] | None = None,
) -> None:
    """Move emigrants to a random destination deme per source.

    Each source population sends ``mig_count`` emigrants to one
    destination chosen uniformly among the other demes. When only one
    deme exists, this is a no-op. Otherwise the placement rules match
    ``mig_ring``. Populations are modified in place.

    Args:
        populations: Populations to migrate between.
        mig_count: Number of individuals to migrate from each population.
        selection: Callable that selects emigrants from a population.
        replacement: Callable that selects which destination individuals
            are replaced. If omitted, the destination's own emigrant
            slots are the vacancies.
    """
    nbr_demes = len(populations)
    if nbr_demes < 2:
        return
    choices = list(range(nbr_demes))
    mig_indices = [int(rng.choice([j for j in choices if j != i])) for i in choices]
    mig_ring(populations, mig_count, selection, replacement, mig_indices=mig_indices)


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
    same object is not left in two demes. A duplicate emigrant
    already present in the destination is cloned so two dest
    slots do not share one object. Populations are modified
    in place.

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

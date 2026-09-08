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
from numbers import Integral
from typing import TYPE_CHECKING, Literal

import numpy

if TYPE_CHECKING:
    from deap_er.private.records.case_exam import CaseExam
    from deap_er.private.typedefs import Individual
from deap_er.private.various.rng import rng

from .sample_informed_cases import sample_informed_cases

__all__: list[str] = ["next_downsample_cases"]

type DownsampleMode = Literal["random", "informed", "cohort", "held_out"]


def next_downsample_cases(
    individuals: list[Individual],
    case_count: int,
    generation: int,
    *,
    mode: DownsampleMode = "random",
    cohort: Sequence[int] | None = None,
    cohorts: Sequence[Sequence[int]] | None = None,
    held_out: CaseExam | None = None,
    matrix: numpy.ndarray | None = None,
    trust_matrix: bool = False,
) -> list[int]:
    """Return the next ``cases=`` list for lexicase down-sampling.

    Chronological meaning stays on the caller. This helper only picks
    catalog indices for one generation.

    Args:
        individuals: Evaluated population supplying fitness length.
        case_count: Target subset size. Values above the catalog are
            capped. ``case_count <= 0`` returns ``[]``.
        generation: Generation index used to rotate cohorts or
            held-out windows.
        mode: ``random``, ``informed``, ``cohort``, or ``held_out``.
        cohort: Fixed case indices for ``mode="cohort"``. Must supply at
            least ``case_count`` distinct valid indices.
        cohorts: Rotating cohort lists for ``mode="cohort"``. When
            both ``cohort`` and ``cohorts`` are set, ``cohort`` wins.
            Each cohort must supply at least ``case_count`` distinct
            valid indices when selected.
        held_out: Caller-marked exam for ``mode="held_out"``.
        matrix: Optional ``(n_individuals, n_cases)`` pack for
            informed mode.
        trust_matrix: When ``True``, ``matrix`` is accepted on shape
            alone. Defaults to ``False``.

    Returns:
        Distinct fitness-case indices for the next lexicase call.

    Raises:
        ValueError: If ``individuals`` is empty, ``case_count`` is not
            an integer, or a mode-specific argument is missing.
    """
    if isinstance(case_count, bool) or not isinstance(case_count, Integral):
        raise ValueError("case_count must be an int")
    count = int(case_count)
    if not individuals:
        raise ValueError("individuals must be non-empty")
    if count <= 0:
        return []
    n_cases = len(individuals[0].fitness.values)
    if n_cases == 0:
        raise ValueError("every individual must have a valid fitness of the same length")
    size = min(count, n_cases)
    if mode == "informed":
        return sample_informed_cases(
            individuals,
            size,
            matrix=matrix,
            trust_matrix=trust_matrix,
        )
    if mode == "cohort":
        return _cohort_cases(cohort, cohorts, generation, size, n_cases)
    if mode == "held_out":
        return _held_out_cases(held_out, generation, size, n_cases)
    return _random_cases(size, n_cases)


def _random_cases(size: int, n_cases: int) -> list[int]:
    pool = list(range(n_cases))
    return list(rng.sample(pool, size))


def _cohort_cases(
    cohort: Sequence[int] | None,
    cohorts: Sequence[Sequence[int]] | None,
    generation: int,
    size: int,
    n_cases: int,
) -> list[int]:
    if cohort is not None:
        source = list(cohort)
    elif cohorts is not None and len(cohorts) > 0:
        source = list(cohorts[int(generation) % len(cohorts)])
    else:
        raise ValueError('mode="cohort" requires cohort or non-empty cohorts')
    if not source:
        raise ValueError("cohort must be non-empty")
    seen: set[int] = set()
    chosen: list[int] = []
    for idx in source:
        value = int(idx)
        if value < 0 or value >= n_cases:
            raise IndexError(f"case index {value} is out of range for {n_cases} fitness cases")
        if value not in seen:
            seen.add(value)
            chosen.append(value)
        if len(chosen) >= size:
            break
    if len(chosen) < size:
        raise ValueError(
            f"cohort must supply at least {size} distinct case indices, got {len(chosen)}"
        )
    return chosen


def _held_out_cases(
    held_out: CaseExam | None,
    generation: int,
    size: int,
    n_cases: int,
) -> list[int]:
    if held_out is None:
        raise ValueError('mode="held_out" requires held_out')
    catalog = held_out.as_cases(n_cases)
    if not catalog:
        raise ValueError("held_out must select at least one case")
    if size >= len(catalog):
        return catalog[:size]
    start = (int(generation) * size) % len(catalog)
    chosen: list[int] = []
    offset = 0
    while len(chosen) < size:
        idx = catalog[(start + offset) % len(catalog)]
        if idx not in chosen:
            chosen.append(idx)
        offset += 1
        if offset > len(catalog):
            break
    return chosen

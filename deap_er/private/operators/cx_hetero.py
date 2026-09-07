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

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual, Mates

__all__: list[str] = ["cx_heterogeneous"]

_SHAPE = "crossovers must be either one callable per gene or (slice, callable) pairs"
_STEP = "crossover slices must be contiguous (step must be 1)"
_OVERLAP = "crossover slices must not overlap"
_UNIT_LEN = "slice crossover must preserve unit length"
_GENE_PAIR = "per-gene crossover must return two replacements"
_SLICE_PAIR = "slice crossover must return two sequences"


def _is_slice_unit(item: object) -> bool:
    """Return whether ``item`` is a ``(slice, callable)`` pair."""
    return isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], slice)


def _require_same_length(ind1: Individual, ind2: Individual) -> int:
    """Return the shared length of two individuals.

    Args:
        ind1: The first individual.
        ind2: The second individual.

    Returns:
        The common sequence length.

    Raises:
        ValueError: If the individuals have different lengths.
    """
    n1, n2 = len(ind1), len(ind2)
    if n1 != n2:
        raise ValueError(f"individuals must have the same length: {n1} != {n2}")
    return n1


def _apply_genes(
    ind1: Individual,
    ind2: Individual,
    crossovers: Sequence[Callable[[Any, Any], tuple[Any, Any]]],
) -> None:
    """Replace each gene with the pair returned by its callable.

    Args:
        ind1: The first individual.
        ind2: The second individual.
        crossovers: One ``(v1, v2) -> (v1', v2')`` callable per gene.

    Raises:
        ValueError: If a callable is a slice spec, is not callable, or
            does not return two replacements.
    """
    for i, crossover in enumerate(crossovers):
        if _is_slice_unit(crossover):
            raise ValueError(_SHAPE)
        if not callable(crossover):
            raise ValueError("each per-gene crossover must be callable")
        result = crossover(ind1[i], ind2[i])
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError(_GENE_PAIR)
        ind1[i], ind2[i] = result


def _resolved_span(slc: slice, size: int) -> tuple[int, int]:
    """Resolve a slice to a contiguous ``[start, stop)`` span.

    Args:
        slc: Slice describing one crossover unit.
        size: Shared individual length.

    Returns:
        Inclusive-start, exclusive-stop indices.

    Raises:
        ValueError: If the slice step is not 1.
    """
    start, stop, step = slc.indices(size)
    if step != 1:
        raise ValueError(_STEP)
    return start, stop


def _apply_slices(
    ind1: Individual,
    ind2: Individual,
    crossovers: Sequence[tuple[slice, Callable[..., Any]]],
    size: int,
) -> None:
    """Run each existing ``cx_*`` on its slice and write the unit back.

    Args:
        ind1: The first individual.
        ind2: The second individual.
        crossovers: ``(slice, cx_*)`` pairs. Slices must be contiguous
            and disjoint. ``cx_es_*`` is unsupported unless the extracted
            unit already carries ``strategy``.
        size: Shared individual length.

    Raises:
        ValueError: If a spec is mixed or invalid, slices overlap, a
            callable changes unit length, or the return is not a pair.
    """
    occupied: list[tuple[int, int]] = []
    for item in crossovers:
        if not _is_slice_unit(item):
            raise ValueError(_SHAPE)
        slc, crossover = item
        if not callable(crossover):
            raise ValueError("each slice crossover must be callable")
        start, stop = _resolved_span(slc, size)
        if any(start < other_stop and other_start < stop for other_start, other_stop in occupied):
            raise ValueError(_OVERLAP)
        occupied.append((start, stop))
        unit_len = stop - start
        part1 = ind1[start:stop]
        part2 = ind2[start:stop]
        result = crossover(part1, part2)
        if result is not None:
            if not isinstance(result, tuple) or len(result) != 2:
                raise ValueError(_SLICE_PAIR)
            part1, part2 = result
        if len(part1) != unit_len or len(part2) != unit_len:
            raise ValueError(_UNIT_LEN)
        ind1[start:stop] = part1
        ind2[start:stop] = part2


def cx_heterogeneous(
    ind1: Individual,
    ind2: Individual,
    crossovers: Sequence[Any],
) -> Mates:
    """Mate two mixed-encoding individuals with per-gene or per-slice operators.

    Both individuals are modified in place. ``crossovers`` is either one
    callable per gene or a sequence of ``(slice, callable)`` pairs, not
    a mix of the two.

    In the per-gene form each callable receives the pair of gene values
    and must return the two replacements. In the per-slice form each
    callable is an existing ``cx_*`` operator: it receives the extracted
    units, may mutate them in place, and should return the two
    replacements. Open slices such as ``slice(3, None)`` resolve against
    the individual length. Uncovered genes are left unchanged.

    Args:
        ind1: The first individual.
        ind2: The second individual.
        crossovers: Per-gene ``(v1, v2) -> (v1', v2')`` callables, or
            ``(slice, cx_*)`` pairs for contiguous blocks.

    Returns:
        The two individuals after crossover.

    Raises:
        ValueError: If the individuals have different lengths, the
            per-gene list length does not match, the two shapes are
            mixed, a slice is not contiguous, slices overlap, a
            callable does not return a pair, or a slice operator
            changes the unit length.
    """
    size = _require_same_length(ind1, ind2)
    if crossovers and _is_slice_unit(crossovers[0]):
        _apply_slices(ind1, ind2, crossovers, size)
    else:
        if len(crossovers) != size:
            raise ValueError(
                "crossovers must have the same length as the individual: "
                f"{len(crossovers)} != {size}"
            )
        _apply_genes(ind1, ind2, crossovers)
    return ind1, ind2

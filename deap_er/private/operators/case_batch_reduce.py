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
from numbers import Integral

import numpy

from deap_er.private.various.rng import rng

__all__: list[str] = [
    "batch_case_matrix",
    "partition_case_batches",
    "reduce_case_mean",
    "reduce_case_mse",
]

type CaseReduction = Callable[[numpy.ndarray], numpy.ndarray]


def reduce_case_mean(block: numpy.ndarray) -> numpy.ndarray:
    """Mean of case columns along axis 1.

    Args:
        block: ``(n_individuals, n_cases)`` values.

    Returns:
        One scalar per individual.
    """
    return numpy.mean(block, axis=1)


def reduce_case_mse(block: numpy.ndarray) -> numpy.ndarray:
    """Mean squared case values along axis 1.

    Args:
        block: ``(n_individuals, n_cases)`` values.

    Returns:
        One scalar per individual.
    """
    return numpy.mean(block * block, axis=1)


def partition_case_batches(subset: Sequence[int], batch_size: int) -> list[list[int]]:
    """Shuffle ``subset`` and split it into consecutive batches.

    Args:
        subset: Fitness-case indices to batch.
        batch_size: Maximum cases per batch. Must be at least ``1``.

    Returns:
        Batches in shuffled order. The last batch may be shorter.

    Raises:
        ValueError: If ``batch_size`` is not a positive integer.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise ValueError("batch_size must be a positive int")
    size = int(batch_size)
    if size < 1:
        raise ValueError("batch_size must be at least 1")
    order = list(subset)
    rng.shuffle(order)
    return [order[i : i + size] for i in range(0, len(order), size)]


def batch_case_matrix(
    matrix: numpy.ndarray,
    subset: list[int],
    fit_weights: tuple[float, ...],
    batch_size: int,
    reduction: CaseReduction,
) -> tuple[numpy.ndarray, tuple[float, ...]]:
    """Collapse shuffled case batches into one column per batch.

    Args:
        matrix: ``(n_individuals, n_cases)`` case matrix.
        subset: Case indices eligible for batching.
        fit_weights: Per-case maximize/minimize signs from fitness.
        batch_size: Maximum cases per batch.
        reduction: Maps a ``(n_individuals, batch_width)`` block to
            ``(n_individuals,)`` batch scores.

    Returns:
        Reduced matrix with shape ``(n_individuals, n_batches)`` and
        the weight copied from the first case in each batch.
    """
    batches = partition_case_batches(subset, batch_size)
    if not batches:
        return numpy.empty((matrix.shape[0], 0), dtype=numpy.float64), ()
    reduced = numpy.empty((matrix.shape[0], len(batches)), dtype=numpy.float64)
    weights: list[float] = []
    for col, batch in enumerate(batches):
        reduced[:, col] = reduction(matrix[:, batch])
        weights.append(fit_weights[batch[0]])
    return reduced, tuple(weights)

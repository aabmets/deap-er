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
from collections.abc import MutableSequence, Sequence
from typing import Any

import numpy

__all__: list[str] = []


def _randint(gen: Any, a: int, b: int) -> int:
    """Return a random integer ``N`` such that ``a <= N <= b``.

    Args:
        gen: NumPy Generator to draw from.
        a: Inclusive lower bound.
        b: Inclusive upper bound.

    Returns:
        An integer in the closed interval.

    Raises:
        ValueError: If the interval is empty.
    """
    return int(gen.integers(a, b, endpoint=True))


def _randrange(gen: Any, start: int, stop: int | None = None, step: int = 1) -> int:
    """Return a random element from ``range(start, stop, step)``.

    Args:
        gen: NumPy Generator to draw from.
        start: Stop if ``stop`` is omitted, otherwise the start.
        stop: Exclusive stop. Optional.
        step: Range step. Defaults to 1.

    Returns:
        An integer from the equivalent ``range`` object.

    Raises:
        ValueError: If the range is empty.
    """
    if stop is None:
        start, stop = 0, start
    values = range(start, stop, step)
    n = len(values)
    if n == 0:
        raise ValueError("empty range for randrange()")
    return values[int(gen.integers(0, n))]


def _choice[T](gen: Any, seq: Sequence[T]) -> T:
    """Return one element of ``seq``.

    Args:
        gen: NumPy Generator to draw from.
        seq: Non-empty sequence. Indexed as a Python sequence.

    Returns:
        One element of ``seq``.

    Raises:
        IndexError: If ``seq`` is empty.
    """
    n = len(seq)
    if n == 0:
        raise IndexError("Cannot choose from an empty sequence")
    return seq[int(gen.integers(0, n))]


def _sample[T](gen: Any, population: Sequence[T], k: int) -> list[T]:
    """Return ``k`` unique elements from ``population``.

    Args:
        gen: NumPy Generator to draw from.
        population: Sequence to sample from.
        k: Number of elements. May be passed by keyword.

    Returns:
        A new list of ``k`` elements.

    Raises:
        ValueError: If ``k`` is negative or larger than
            ``len(population)``.
    """
    n = len(population)
    if k < 0 or k > n:
        raise ValueError("Sample larger than population or is negative")
    if k == 0:
        return []
    indices = gen.choice(n, size=k, replace=False)
    return [population[int(i)] for i in indices]


def _shuffle(gen: Any, x: MutableSequence[Any] | numpy.ndarray) -> None:
    """Shuffle ``x`` in place.

    Sequences of individuals are permuted by index. A 1-D ndarray
    is shuffled by the underlying Generator.

    Args:
        gen: NumPy Generator to draw from.
        x: Mutable sequence or ndarray to shuffle.
    """
    if isinstance(x, numpy.ndarray):
        gen.shuffle(x)
        return
    n = len(x)
    if n < 2:
        return
    order = gen.permutation(n)
    shuffled = [x[int(i)] for i in order]
    x[:] = shuffled


def _gauss(gen: Any, mu: float, sigma: float) -> float:
    """Return a sample from a Gaussian distribution.

    Args:
        gen: NumPy Generator to draw from.
        mu: Mean.
        sigma: Standard deviation.

    Returns:
        A Python float from the normal distribution.
    """
    return float(gen.normal(mu, sigma))


def _standard_normal(gen: Any, size: int | tuple[int, ...] | None = None) -> float | numpy.ndarray:
    """Return samples from the standard normal distribution.

    Args:
        gen: NumPy Generator to draw from.
        size: Output shape. A single float is returned when omitted.

    Returns:
        A float, or an ndarray when ``size`` is given.
    """
    if size is None:
        return float(gen.standard_normal())
    return gen.standard_normal(size)


def _integers(
    gen: Any,
    low: int,
    high: int | None = None,
    size: int | tuple[int, ...] | None = None,
    endpoint: bool = False,
) -> int | numpy.ndarray:
    """Return random integers from the underlying Generator.

    Args:
        gen: NumPy Generator to draw from.
        low: Inclusive lower bound, or exclusive high when
            ``high`` is omitted.
        high: Exclusive upper bound unless ``endpoint`` is True.
        size: Output shape. A single int is returned when omitted.
        endpoint: If True, ``high`` is inclusive.

    Returns:
        An int, or an ndarray when ``size`` is given.
    """
    return gen.integers(low, high, size=size, endpoint=endpoint)

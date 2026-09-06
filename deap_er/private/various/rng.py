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
from typing import Any, cast, overload

import numpy

__all__: list[str] = ["RNG", "rng"]

_BUFSIZE = 1024


class RNG:
    """NumPy Generator facade with a buffered uniform stream.

    ``random`` and ``uniform`` pop Python floats from a refillable
    buffer. Other methods draw from the same ``Generator`` directly.
    Populations of individuals are indexed as Python sequences so a
    list of list-genes is never flattened into an array.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Create a generator, optionally seeded.

        Args:
            seed: Seed for the NumPy Generator. Optional; entropy from
                the OS is used when omitted.
        """
        self._buf = numpy.empty(_BUFSIZE, dtype=numpy.float64)
        self._floats: list[float] = []
        self._i = _BUFSIZE
        self._gen = numpy.random.default_rng(seed)

    def seed(self, seed: int | None = None) -> None:
        """Reseed the generator and discard unused buffered uniforms.

        Args:
            seed: Seed for a new NumPy Generator. Optional; entropy
                from the OS is used when omitted.
        """
        self._gen = numpy.random.default_rng(seed)
        self._i = _BUFSIZE

    def get_state(self) -> dict[str, Any]:
        """Return the bit-generator state and unused buffer.

        Returns:
            A mapping with ``bit_generator``, ``buf``, and ``index``.
        """
        return {
            "bit_generator": self._gen.bit_generator.state,
            "buf": self._buf.copy(),
            "index": self._i,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore a state previously returned by ``get_state``.

        Args:
            state: Mapping with ``bit_generator``, ``buf``, and
                ``index``.
        """
        self._gen.bit_generator.state = state["bit_generator"]
        self._buf = numpy.array(state["buf"], dtype=numpy.float64, copy=True)
        self._floats = cast(list[float], self._buf.tolist())
        self._i = int(state["index"])

    def random(self) -> float:
        """Return the next uniform float in ``[0.0, 1.0)``.

        Returns:
            A Python float from the buffered stream.
        """
        if self._i >= _BUFSIZE:
            self._gen.random(out=self._buf)
            self._floats = cast(list[float], self._buf.tolist())
            self._i = 0
        value = self._floats[self._i]
        self._i += 1
        return value

    def uniform(self, a: float, b: float) -> float:
        """Return a uniform float in ``[a, b)``.

        Args:
            a: Lower bound.
            b: Upper bound.

        Returns:
            ``a + (b - a) * random()``.
        """
        return a + (b - a) * self.random()

    def randint(self, a: int, b: int) -> int:
        """Return a random integer ``N`` such that ``a <= N <= b``.

        Args:
            a: Inclusive lower bound.
            b: Inclusive upper bound.

        Returns:
            An integer in the closed interval.

        Raises:
            ValueError: If the interval is empty.
        """
        return int(self._gen.integers(a, b, endpoint=True))

    def randrange(self, start: int, stop: int | None = None, step: int = 1) -> int:
        """Return a random element from ``range(start, stop, step)``.

        Args:
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
        return values[int(self._gen.integers(0, n))]

    def choice[T](self, seq: Sequence[T]) -> T:
        """Return one element of ``seq``.

        Args:
            seq: Non-empty sequence. Indexed as a Python sequence.

        Returns:
            One element of ``seq``.

        Raises:
            IndexError: If ``seq`` is empty.
        """
        n = len(seq)
        if n == 0:
            raise IndexError("Cannot choose from an empty sequence")
        return seq[int(self._gen.integers(0, n))]

    def sample[T](self, population: Sequence[T], k: int) -> list[T]:
        """Return ``k`` unique elements from ``population``.

        Args:
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
        indices = self._gen.choice(n, size=k, replace=False)
        return [population[int(i)] for i in indices]

    def shuffle(self, x: MutableSequence[Any] | numpy.ndarray) -> None:
        """Shuffle ``x`` in place.

        Sequences of individuals are permuted by index. A 1-D ndarray
        is shuffled by the underlying Generator.

        Args:
            x: Mutable sequence or ndarray to shuffle.
        """
        if isinstance(x, numpy.ndarray):
            self._gen.shuffle(x)
            return
        n = len(x)
        if n < 2:
            return
        order = self._gen.permutation(n)
        shuffled = [x[int(i)] for i in order]
        x[:] = shuffled

    def gauss(self, mu: float, sigma: float) -> float:
        """Return a sample from a Gaussian distribution.

        Args:
            mu: Mean.
            sigma: Standard deviation.

        Returns:
            A Python float from the normal distribution.
        """
        return float(self._gen.normal(mu, sigma))

    @overload
    def standard_normal(self, size: None = None) -> float: ...

    @overload
    def standard_normal(self, size: int | tuple[int, ...]) -> numpy.ndarray: ...

    def standard_normal(self, size: int | tuple[int, ...] | None = None) -> float | numpy.ndarray:
        """Return samples from the standard normal distribution.

        Args:
            size: Output shape. A single float is returned when
                omitted.

        Returns:
            A float, or an ndarray when ``size`` is given.
        """
        if size is None:
            return float(self._gen.standard_normal())
        return self._gen.standard_normal(size)

    @overload
    def integers(
        self,
        low: int,
        high: int | None = None,
        size: None = None,
        endpoint: bool = False,
    ) -> int: ...

    @overload
    def integers(
        self,
        low: int,
        high: int | None = None,
        *,
        size: int | tuple[int, ...],
        endpoint: bool = False,
    ) -> numpy.ndarray: ...

    def integers(
        self,
        low: int,
        high: int | None = None,
        size: int | tuple[int, ...] | None = None,
        endpoint: bool = False,
    ) -> int | numpy.ndarray:
        """Return random integers from the underlying Generator.

        Args:
            low: Inclusive lower bound, or exclusive high when
                ``high`` is omitted.
            high: Exclusive upper bound unless ``endpoint`` is True.
            size: Output shape. A single int is returned when omitted.
            endpoint: If True, ``high`` is inclusive.

        Returns:
            An int, or an ndarray when ``size`` is given.
        """
        return self._gen.integers(low, high, size=size, endpoint=endpoint)


rng = RNG()

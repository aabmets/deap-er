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
from typing import Any, cast

import numpy

__all__: list[str] = ["RngBuffers", "draw_integers"]

_BUFSIZE = 1024
_U64_SPACE = 1 << 64


class RngBuffers:
    """Refillable uniform-float and raw-uint64 streams."""

    def __init__(self) -> None:
        """Allocate zeroed buffers that refill on the first pop."""
        self._fbuf = numpy.zeros(_BUFSIZE, dtype=numpy.float64)
        self._floats: list[float] = []
        self._fi = _BUFSIZE
        self._reset_ints()

    def discard(self) -> None:
        """Drop leftover values and zero unused buffer storage."""
        self._fbuf.fill(0)
        self._floats = []
        self._fi = _BUFSIZE
        self._reset_ints()

    def _reset_ints(self) -> None:
        """Forget leftover uint64s and zero the integer array."""
        self._ibuf = numpy.zeros(_BUFSIZE, dtype=numpy.uint64)
        self._u64s = []
        self._ii = _BUFSIZE

    def pack(self) -> dict[str, Any]:
        """Copy leftover buffer arrays and their read indices.

        Spent buffers (index at capacity) are exported as zeros so a
        checkpoint never carries uninitialized or discarded words.

        Returns:
            ``buf``, ``index``, ``ibuf``, and ``iindex``.
        """
        fbuf = self._fbuf.copy()
        ibuf = self._ibuf.copy()
        if self._fi >= _BUFSIZE:
            fbuf.fill(0)
        if self._ii >= _BUFSIZE:
            ibuf.fill(0)
        return {
            "buf": fbuf,
            "index": self._fi,
            "ibuf": ibuf,
            "iindex": self._ii,
        }

    def unpack(self, state: dict[str, Any]) -> None:
        """Restore buffers from ``pack`` or a legacy uniform-only mapping.

        Args:
            state: Mapping with ``buf`` and ``index``. ``ibuf`` and
                ``iindex`` are optional; omit them for an empty
                integer buffer.
        """
        self._fbuf = numpy.array(state["buf"], dtype=numpy.float64, copy=True)
        self._floats = cast(list[float], self._fbuf.tolist())
        self._fi = int(state["index"])
        if "ibuf" in state and "iindex" in state:
            self._ibuf = numpy.array(state["ibuf"], dtype=numpy.uint64, copy=True)
            self._u64s = [int(value) for value in self._ibuf.tolist()]
            self._ii = int(state["iindex"])
            return
        self._reset_ints()

    def next_float(self, gen: numpy.random.Generator) -> float:
        """Pop the next uniform float in ``[0.0, 1.0)``.

        Args:
            gen: Generator used to refill an empty float buffer.

        Returns:
            A Python float from the leftover uniforms.
        """
        if self._fi >= _BUFSIZE:
            gen.random(out=self._fbuf)
            self._floats = cast(list[float], self._fbuf.tolist())
            self._fi = 0
        value = self._floats[self._fi]
        self._fi += 1
        return value

    def next_u64(self, gen: numpy.random.Generator) -> int:
        """Pop the next raw 64-bit word.

        Args:
            gen: Generator whose bit generator fills an empty buffer.

        Returns:
            A Python int in ``[0, 2**64)``.
        """
        if self._ii >= _BUFSIZE:
            self._ibuf = numpy.asarray(gen.bit_generator.random_raw(_BUFSIZE), dtype=numpy.uint64)
            self._u64s = [int(value) for value in self._ibuf.tolist()]
            self._ii = 0
        value = int(self._u64s[self._ii])
        self._ii += 1
        return value

    def offset(self, gen: numpy.random.Generator, start: int, span: int) -> int:
        """Return ``start`` plus an unbiased integer in ``[0, span)``.

        Args:
            gen: Generator used to refill the uint64 buffer.
            start: Inclusive origin of the interval.
            span: Number of integers in the interval.

        Returns:
            An integer in ``[start, start + span)``.

        Raises:
            ValueError: If ``span`` is not positive.
            OverflowError: If ``span`` exceeds ``2**64``.
        """
        return start + self.index(gen, span)

    def index(self, gen: numpy.random.Generator, n: int) -> int:
        """Return an unbiased integer in ``[0, n)``.

        Always consumes at least one uint64, including when ``n`` is 1.

        Args:
            gen: Generator used to refill the uint64 buffer.
            n: Exclusive upper bound. Must be in ``1 .. 2**64``.

        Returns:
            An integer in the half-open interval.

        Raises:
            ValueError: If ``n`` is not positive.
            OverflowError: If ``n`` exceeds ``2**64``.
        """
        if n <= 0:
            raise ValueError("low >= high")
        if n > _U64_SPACE:
            raise OverflowError("high - low is too large for 64-bit sampling")
        limit = _U64_SPACE - (_U64_SPACE % n)
        while True:
            value = self.next_u64(gen)
            if value < limit:
                return value % n


def draw_integers(
    gen: numpy.random.Generator,
    buffers: RngBuffers,
    low: int,
    high: int | None,
    size: int | tuple[int, ...] | None,
    endpoint: bool,
) -> int | numpy.ndarray:
    """Draw a scalar int from the uint64 buffer, or a sized array from ``gen``.

    A sized draw discards leftover uint64s so later scalar ints refill
    from the advanced bit generator. Leftover uniforms are kept.

    Args:
        gen: Underlying NumPy Generator.
        buffers: Float and uint64 leftovers.
        low: Inclusive lower bound, or exclusive high when ``high``
            is omitted.
        high: Exclusive upper bound unless ``endpoint`` is True.
        size: Output shape. A single int is returned when omitted.
        endpoint: If True, ``high`` is inclusive.

    Returns:
        An int, or an ndarray when ``size`` is given.

    Raises:
        ValueError: If the interval is empty.
        OverflowError: If the span exceeds 64 bits.
    """
    if size is not None:
        buffers._reset_ints()
        return gen.integers(low, high, size=size, endpoint=endpoint)
    if high is None:
        low, high = 0, low
    span = high - low + 1 if endpoint else high - low
    return buffers.offset(gen, low, span)

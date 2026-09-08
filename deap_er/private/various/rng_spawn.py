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
import operator
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy

from .rng import RNG, rng

__all__: list[str] = [
    "bind_spawned_rng",
    "call_spawned",
    "map_spawned",
    "spawn_rng",
]

type SpawnedPayload[T] = tuple[int | Sequence[int], int, Callable[[T], Any], T]


def _entropy(seed: int | Sequence[int]) -> int | list[int]:
    """Normalize a run seed for ``SeedSequence``.

    Args:
        seed: Integer seed or a non-empty sequence of integers.

    Returns:
        An ``int`` or a list of ``int`` values.

    Raises:
        TypeError: If ``seed`` is not an int or a sequence of ints.
        ValueError: If a sequence seed is empty.
    """
    if isinstance(seed, (str, bytes, bytearray)):
        raise TypeError("seed must be an int or a sequence of ints")
    if isinstance(seed, Sequence):
        values = [operator.index(part) for part in seed]
        if not values:
            raise ValueError("seed sequence must not be empty")
        return values
    return operator.index(seed)


def _worker_index(worker_id: int) -> int:
    """Return a non-negative integer worker id.

    Args:
        worker_id: Stable task index. Numpy integers are accepted.

    Returns:
        ``worker_id`` as a Python ``int``.

    Raises:
        TypeError: If ``worker_id`` is not an integer.
        ValueError: If ``worker_id`` is negative.
    """
    try:
        index = operator.index(worker_id)
    except TypeError:
        raise TypeError("worker_id must be an integer") from None
    if index < 0:
        raise ValueError("worker_id must be non-negative")
    return index


def spawn_rng(seed: int | Sequence[int], worker_id: int) -> RNG:
    """Return an independent RNG derived from ``seed`` and ``worker_id``.

    Uses a NumPy ``SeedSequence`` child of the run seed. The process-wide
    generator is not read or advanced, so a parent seeded with the same
    value keeps its own stream. The same pair always yields the same
    stream, regardless of which process draws it or when.

    Args:
        seed: Run seed. The same value passed to ``rng.seed``.
        worker_id: Stable non-negative task index, not an OS pid.

    Returns:
        A new ``RNG`` that does not share state with the parent.

    Raises:
        TypeError: If ``seed`` or ``worker_id`` has a bad type.
        ValueError: If ``seed`` is an empty sequence or ``worker_id``
            is negative.
    """
    spawn_key = (_worker_index(worker_id),)
    sequence = numpy.random.SeedSequence(_entropy(seed), spawn_key=spawn_key)
    mixed = 0
    for word in sequence.generate_state(4):
        mixed = (mixed << 32) | int(word)
    return RNG(mixed)


def bind_spawned_rng(seed: int | Sequence[int], worker_id: int) -> RNG:
    """Install ``spawn_rng(seed, worker_id)`` as the process-wide generator.

    Args:
        seed: Run seed. The same value passed to ``rng.seed``.
        worker_id: Stable non-negative task index.

    Returns:
        The process-wide ``rng`` singleton, now on the spawned stream.
    """
    rng.set_state(spawn_rng(seed, worker_id).get_state())
    return rng


def call_spawned[T](payload: SpawnedPayload[T]) -> tuple[int, Any]:
    """Bind a spawned stream, call ``func(item)``, then restore ``rng``.

    The process-wide generator is restored even if ``func`` raises, so an
    in-process map does not leak a worker stream into the parent.

    Args:
        payload: ``(seed, worker_id, func, item)``.

    Returns:
        ``(worker_id, func(item))`` so results can be ordered by id.
    """
    seed, worker_id, func, item = payload
    previous = rng.get_state()
    try:
        bind_spawned_rng(seed, worker_id)
        return worker_id, func(item)
    finally:
        rng.set_state(previous)


def map_spawned[T, R](
    func: Callable[[T], R],
    iterable: Iterable[T],
    *,
    seed: int | Sequence[int],
    map_func: Callable[..., Iterable[Any]] = map,
) -> list[R]:
    """Map ``func`` over ``iterable`` with one spawned stream per item.

    Item ``i`` sees ``spawn_rng(seed, i)`` as the process-wide ``rng``.
    Results are returned in input order. ``map_func`` may complete
    tasks in any order; only the ``worker_id`` on each result matters.

    Args:
        func: Callable applied to each item. Library operators that
            read ``rng`` see the spawned stream.
        iterable: Inputs. Materialized once so ids are stable.
        seed: Run seed shared with ``rng.seed`` on the parent.
        map_func: ``map``-like callable. Defaults to builtin ``map``.
            Pass ``pool.map`` or an unordered mapper.

    Returns:
        ``func`` results in the same order as ``iterable``.

    Raises:
        ValueError: If ``map_func`` does not return one result per item.
    """
    items = list(iterable)
    if not items:
        return []
    payloads: list[SpawnedPayload[T]] = [
        (seed, index, func, item) for index, item in enumerate(items)
    ]
    ordered = sorted(map_func(call_spawned, payloads), key=lambda pair: pair[0])
    ids = [pair[0] for pair in ordered]
    if ids != list(range(len(items))):
        raise ValueError("map_func must return one (worker_id, value) per item")
    return [pair[1] for pair in ordered]

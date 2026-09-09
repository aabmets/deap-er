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

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual
    from deap_er.private.various.eval_cache import EvalCache

__all__: list[str] = ["noisy_draw_key", "resample", "resample_aggregate"]


def noisy_draw_key(base: Any, draw: int) -> tuple[Any, int]:
    """Return a hashable :class:`EvalCache` caller key that includes the draw.

    A noisy ``evaluate`` must vary the cache key per draw. Without a
    per-draw key, ``EvalCache`` returns the first draw for every repeat.

    Args:
        base: Caller-owned expression or matrix identity fragment.
        draw: Zero-based resample index.

    Returns:
        ``(base, draw)`` suitable for ``EvalCache.evaluate(..., key=)``.
    """
    return (base, int(draw))


def resample_aggregate(samples: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Average per-objective samples, skipping non-finite values.

    Args:
        samples: Fitness tuples from independent draws.

    Returns:
        Elementwise means across ``samples``.

    Raises:
        ValueError: If ``samples`` is empty or tuple lengths differ.
    """
    if not samples:
        raise ValueError("samples must not be empty")
    width = len(samples[0])
    for sample in samples:
        if len(sample) != width:
            raise ValueError("all samples must have the same length")
    means: list[float] = []
    for objective in range(width):
        values = [
            float(sample[objective]) for sample in samples if math.isfinite(sample[objective])
        ]
        if not values:
            means.append(float("nan"))
        else:
            means.append(sum(values) / len(values))
    return tuple(means)


def resample(
    ind: Individual,
    evaluate: Callable[[Individual], Sequence[float]],
    n: int,
    *,
    cache: EvalCache | None = None,
    key: Any = None,
    aggregate: Callable[[Sequence[Sequence[float]]], Sequence[float]] = resample_aggregate,
    write: bool = True,
) -> tuple[float, ...]:
    """Score one individual ``n`` times and aggregate the noisy draws.

    Each repeat goes through ``cache`` when it is set. Pass
    :func:`noisy_draw_key` (or an equivalent draw-specific key) so
    independent draws do not share one cache entry.

    Args:
        ind: Individual to score.
        evaluate: Callable that returns one fitness tuple per draw.
        n: Number of independent draws. Must be at least ``1``.
        cache: Optional :class:`~deap_er.tools.EvalCache` wrapper.
        key: Optional caller key fragment paired with each draw index.
        aggregate: Reduces draw tuples to one fitness tuple.
        write: When true, assign the aggregate to ``ind.fitness.values``.

    Returns:
        The aggregated fitness tuple.

    Raises:
        ValueError: If ``n`` is less than ``1``.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    samples: list[Sequence[float]] = []
    for draw in range(n):
        if cache is not None:
            draw_key = noisy_draw_key(key, draw) if key is not None else draw
            sample = cache.evaluate(ind, key=draw_key)
        else:
            sample = evaluate(ind)
        samples.append(tuple(float(value) for value in sample))
    values = tuple(aggregate(samples))
    if write:
        ind.fitness.values = values
    return values

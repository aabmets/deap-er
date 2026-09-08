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
from typing import Any
from weakref import WeakSet

from deap_er.private.programming.compile_cache import expression_key

__all__: list[str] = ["EvalCache", "clear_eval_caches", "invalidate_eval"]

CacheKey = tuple[Any, int | None, int]
_eval_caches: WeakSet[EvalCache] = WeakSet()


def clear_eval_caches() -> None:
    """Drop every entry from each live :class:`EvalCache`.

    ``clear_compile_cache`` calls this so ``promote_subtree`` (and any
    other language mutation) cannot leave stale fitness next to a
    dropped compile-cache table.
    """
    # Mutate cache entries only; the WeakSet itself is not updated.
    for cache in _eval_caches:
        cache.clear()


def invalidate_eval(expr: Any) -> int:
    """Drop matching expression keys from every live :class:`EvalCache`.

    Uses the same fragment as ``invalidate_compiled``: a structural
    tree key, raw source text, or the historical ``lambda …: {expr}``
    form. ``tune_ephemerals`` reaches this through
    ``invalidate_compiled``.

    Args:
        expr: Expression whose cached fitness should be evicted.

    Returns:
        The number of cache entries removed across all live caches.
    """
    fragment = expr if isinstance(expr, str | tuple) else expression_key(expr)
    return sum(cache.invalidate(fragment) for cache in _eval_caches)


def _expression_fragment(individual: Any, caller_key: Any) -> Any:
    """Return the expression half of an eval-cache key."""
    if caller_key is not None:
        return caller_key
    if isinstance(individual, str):
        return individual
    try:
        node = individual[0]
    except (TypeError, IndexError, KeyError):
        return str(individual)
    if getattr(node, "arity", None) is None or getattr(node, "name", None) is None:
        return str(individual)
    return expression_key(individual)


def _matrix_part(matrix: Any) -> tuple[int | None, int]:
    """Return ``(id(matrix), row_count)`` for the cache key."""
    if matrix is None:
        return None, 0
    if hasattr(matrix, "shape") and getattr(matrix, "shape", ()):
        return id(matrix), int(matrix.shape[0])
    try:
        return id(matrix), len(matrix)
    except TypeError:
        return id(matrix), 0


def _fragment_matches(code: Any, expression: Any) -> bool:
    """Return whether a stored expression fragment names ``expression``."""
    if code == expression:
        return True
    if not isinstance(code, str) or not isinstance(expression, str):
        return False
    return code.endswith(f": {expression}")


class EvalCache:
    """Fitness cache keyed by expression plus matrix identity / rows.

    Wrap the caller's ``evaluate`` / ``evaluate_batch``. A hit returns
    the stored fitness tuple and does not call the wrapped callable.
    The default expression key is tree structure via
    ``expression_key``, source text, or ``str(individual)``. Pass
    ``key=`` (or ``keys=`` on a batch) to override.

    Live instances register with the process-wide invalidation hook:
    ``clear_compile_cache`` clears every cache; ``invalidate_compiled``
    drops keys for that expression. ``promote_subtree`` and
    ``tune_ephemerals`` already call those helpers, so a language
    mutation or ephemeral write-back cannot keep a stale fitness.

    ``n_evals`` and logbook ``nevals`` still count every fitness
    assignment through ``evaluate_invalid``. A cache hit skips the
    wrapped callable only.

    Args:
        evaluate: Optional ``callable(ind) ->`` fitness tuple.
        evaluate_batch: Optional ``callable(inds) ->`` fitness tuples.
        matrix: Evaluation matrix whose identity and row count join
            the key. ``None`` stores ``(None, 0)``.
        key_fn: Optional ``callable(ind) ->`` hashable caller key.
    """

    def __init__(
        self,
        evaluate: Callable[[Any], Any] | None = None,
        evaluate_batch: Callable[[list[Any]], Any] | None = None,
        *,
        matrix: Any = None,
        key_fn: Callable[[Any], Any] | None = None,
    ) -> None:
        """See the class docstring."""
        self._evaluate = evaluate
        self._evaluate_batch = evaluate_batch
        self.matrix = matrix
        self._key_fn = key_fn
        self._entries: dict[CacheKey, Any] = {}
        _eval_caches.add(self)

    def cache_key(self, individual: Any, caller_key: Any = None) -> CacheKey:
        """Build the key for ``individual`` on the current matrix.

        Args:
            individual: Individual or expression being scored.
            caller_key: Optional explicit key. Overrides ``key_fn``.

        Returns:
            ``(expression, id(matrix), row_count)``.
        """
        override = caller_key
        if override is None and self._key_fn is not None:
            override = self._key_fn(individual)
        ident, rows = _matrix_part(self.matrix)
        return _expression_fragment(individual, override), ident, rows

    def evaluate(self, individual: Any, *, key: Any = None) -> Any:
        """Return cached fitness or pay ``evaluate`` on a miss.

        Args:
            individual: Individual to score.
            key: Optional caller key for this call.

        Returns:
            The fitness tuple from the cache or from ``evaluate``.

        Raises:
            ValueError: If no ``evaluate`` callable was given.
        """
        cache_key = self.cache_key(individual, key)
        if cache_key in self._entries:
            return self._entries[cache_key]
        if self._evaluate is None:
            raise ValueError("EvalCache.evaluate requires an evaluate callable.")
        fitness = self._evaluate(individual)
        self._entries[cache_key] = fitness
        return fitness

    def evaluate_batch(
        self,
        individuals: Sequence[Any],
        *,
        keys: Sequence[Any] | None = None,
    ) -> list[Any]:
        """Score a batch, calling ``evaluate_batch`` only for misses.

        When ``evaluate_batch`` is omitted, each miss uses ``evaluate``.

        Args:
            individuals: Individuals to score, in caller order.
            keys: Optional per-individual caller keys.

        Returns:
            Fitness tuples aligned with ``individuals``.

        Raises:
            ValueError: If neither wrapped callable is set.
            ValueError: If ``keys`` is given and its length differs.
        """
        if keys is not None and len(keys) != len(individuals):
            raise ValueError("keys must have one entry per individual.")
        results: list[Any] = [None] * len(individuals)
        misses: list[Any] = []
        miss_idx: list[int] = []
        for index, individual in enumerate(individuals):
            caller_key = None if keys is None else keys[index]
            cache_key = self.cache_key(individual, caller_key)
            if cache_key in self._entries:
                results[index] = self._entries[cache_key]
                continue
            misses.append(individual)
            miss_idx.append(index)
        if not misses:
            return results
        fresh = list(self._score_misses(misses))
        for index, fitness, individual in zip(miss_idx, fresh, misses, strict=True):
            caller_key = None if keys is None else keys[index]
            self._entries[self.cache_key(individual, caller_key)] = fitness
            results[index] = fitness
        return results

    def invalidate(self, expr: Any) -> int:
        """Drop entries whose expression fragment names ``expr``.

        Args:
            expr: ``expression_key`` fragment, source text, or tree.

        Returns:
            The number of entries removed.
        """
        fragment = expr if isinstance(expr, str | tuple) else expression_key(expr)
        drop = [key for key in self._entries if _fragment_matches(key[0], fragment)]
        for key in drop:
            self._entries.pop(key, None)
        return len(drop)

    def clear(self) -> None:
        """Remove every cached fitness tuple."""
        self._entries.clear()

    def __len__(self) -> int:
        """Return how many fitness tuples the cache currently holds."""
        return len(self._entries)

    def _score_misses(self, misses: list[Any]) -> Any:
        """Evaluate cache misses through the batch or scalar path."""
        if self._evaluate_batch is not None:
            return self._evaluate_batch(misses)
        if self._evaluate is not None:
            return [self._evaluate(individual) for individual in misses]
        raise ValueError("EvalCache.evaluate_batch requires evaluate or evaluate_batch.")

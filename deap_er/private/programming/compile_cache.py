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

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

__all__: list[str] = ["CompileCache", "compile_cache_key", "expression_key"]

CacheKey = tuple[Any, ...]


def expression_key(expr: Any) -> str | tuple[Any, ...]:
    """Return the cache fragment that identifies ``expr``.

    Source text is stored as-is. A tree is stored as
    ``(name, value, arity, call_zero)`` per node so a cache hit does
    not have to render Python. An unhashable leaf or a non-iterable
    expression falls back to ``str(expr)``.

    Args:
        expr: Source text, a prefix-ordered tree, or another object
            whose string form is valid Python.

    Returns:
        A hashable key fragment for ``compile_tree``.
    """
    if isinstance(expr, str):
        return expr
    try:
        key = tuple(
            (
                getattr(node, "name", None),
                getattr(node, "value", None),
                getattr(node, "arity", 0),
                getattr(node, "call_zero", False),
            )
            for node in expr
        )
        hash(key)
        return key
    except TypeError:
        return str(expr)


def compile_cache_key(
    backend: str,
    dispatch: Any,
    expr: Any,
    arguments: Sequence[str],
    context: Mapping[str, Any],
    generation: int,
) -> CacheKey:
    """Build the process-wide compile-cache key for one expression.

    Args:
        backend: Compile backend name.
        dispatch: Numba consumer kernel, or None.
        expr: Expression passed to ``compile_tree``.
        arguments: Primitive-set argument names, in order.
        context: Evaluation context whose value identities matter.
        generation: Promoted-library generation, or 0.

    Returns:
        The LRU key used by ``compile_tree``.
    """
    ctx_key = tuple(sorted((name, id(value)) for name, value in context.items()))
    return (
        backend,
        id(dispatch),
        expression_key(expr),
        tuple(arguments),
        ctx_key,
        generation,
    )


class CompileCache:
    """Least-recently-used cache for compiled GP expressions.

    Evicts one entry at a time when the cache is full so a diverse
    working set is not flushed together.
    """

    def __init__(self, maxsize: int = 1024) -> None:
        """Create a cache with a fixed capacity.

        Args:
            maxsize: Maximum number of compiled callables to retain.
        """
        self._maxsize = maxsize
        self._entries: OrderedDict[CacheKey, Any] = OrderedDict()

    def get(self, key: CacheKey) -> Any | None:
        """Return a cached value and mark it recently used.

        Args:
            key: Cache key produced by ``compile_tree``.

        Returns:
            The cached callable, or ``None`` when the key is absent.
        """
        value = self._entries.pop(key, None)
        if value is None:
            return None
        self._entries[key] = value
        return value

    def set(self, key: CacheKey, value: Any) -> None:
        """Store a compiled value, evicting the oldest entry when full.

        Args:
            key: Cache key produced by ``compile_tree``.
            value: Compiled callable or constant result to retain.
        """
        if key in self._entries:
            self._entries.pop(key)
        elif len(self._entries) >= self._maxsize:
            self._entries.popitem(last=False)
        self._entries[key] = value

    def clear(self) -> None:
        """Remove every cached entry."""
        self._entries.clear()

    def discard_expression(self, expression: Any) -> int:
        """Drop entries whose fragment is ``expression`` or wraps it.

        ``compile_tree`` stores a structural tree key, raw source text,
        or (historically) ``lambda args: {expression}``. All three
        forms are removed when they name ``expression``.

        Args:
            expression: ``expression_key`` fragment or source text.

        Returns:
            The number of entries removed.
        """
        drop = [
            key
            for key in self._entries
            if _fragment_matches(key[2] if len(key) > 2 else None, expression)
        ]
        for key in drop:
            self._entries.pop(key, None)
        return len(drop)

    def __len__(self) -> int:
        """Return how many entries the cache currently holds."""
        return len(self._entries)


def _fragment_matches(code: Any, expression: Any) -> bool:
    """Return whether a cache-key fragment names ``expression``."""
    if code == expression:
        return True
    if not isinstance(code, str) or not isinstance(expression, str):
        return False
    return code.endswith(f": {expression}")

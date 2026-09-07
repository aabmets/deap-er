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
from typing import Any

__all__: list[str] = ["CompileCache"]

CacheKey = tuple[Any, ...]


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

    def discard_expression(self, expression: str) -> int:
        """Drop entries whose code is ``expression`` or a lambda wrapping it.

        ``compile_tree`` stores either the raw expression text or
        ``lambda args: {expression}``. Both forms are removed.

        Args:
            expression: ``str`` of the tree or expression.

        Returns:
            The number of entries removed.
        """
        suffix = f": {expression}"
        drop = [
            key
            for key in self._entries
            if _code_matches(key[2] if len(key) > 2 else None, expression, suffix)
        ]
        for key in drop:
            self._entries.pop(key, None)
        return len(drop)

    def __len__(self) -> int:
        """Return how many entries the cache currently holds."""
        return len(self._entries)


def _code_matches(code: Any, expression: str, suffix: str) -> bool:
    """Return whether a cache-key code component names ``expression``."""
    return isinstance(code, str) and (code == expression or code.endswith(suffix))

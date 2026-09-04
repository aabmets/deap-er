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
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any

__all__ = ["Statistics", "MultiStatistics"]


class Statistics:
    """Compile named statistics on a sequence of objects.

    ``key`` selects the value scored on each element. The default key
    is the identity function. The key may return a sequence when the
    registered functions accept one, for example a multi-objective
    fitness passed to a NumPy statistic.

    Args:
        key: Extracts the value to score from each element. Defaults
            to the identity function.
    """

    def __init__(self, key: Callable[..., Any] | None = None) -> None:
        """See the class docstring."""
        self.key = key if key else lambda obj: obj
        self.functions = {}
        self.fields = []

    def register(self, name: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Register a statistic computed by ``compile``.

        Extra positional and keyword arguments are bound into ``func``.

        Args:
            name: Key used for this statistic in the compiled record.
            func: Function applied to the sequence of key values.
            *args: Positional arguments bound into ``func``.
            **kwargs: Keyword arguments bound into ``func``.
        """
        self.functions[name] = partial(func, *args, **kwargs)
        self.fields.append(name)

    def compile(self, data: Iterable[Any]) -> dict[str, Any]:
        """Compute every registered statistic on ``data``.

        Args:
            data: Iterable of elements passed through ``key``.

        Returns:
            Mapping of registered names to computed values.
        """
        entry = {}
        values = tuple(self.key(elem) for elem in data)
        for key, func in self.functions.items():
            entry[key] = func(values)
        return entry


class MultiStatistics(dict[str, Any]):
    """Compile several named ``Statistics`` objects in one call.

    Construct with keyword arguments that map a chapter name to a
    ``Statistics`` instance, for example
    ``MultiStatistics(fitness=stats_fit, size=stats_size)``.
    ``register`` forwards the same function to every chapter.
    """

    @property
    def fields(self) -> list[str]:
        """Sorted names of the contained ``Statistics`` objects."""
        return sorted(self.keys())

    def register(self, name: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Register ``func`` on every contained ``Statistics`` object.

        Args:
            name: Key used for this statistic in each chapter record.
            func: Function applied to each chapter's key values.
            *args: Positional arguments bound into ``func``.
            **kwargs: Keyword arguments bound into ``func``.
        """
        for stats in self.values():
            stats.register(name, func, *args, **kwargs)

    def compile(self, data: Iterable[Any]) -> dict[str, Any]:
        """Compile every contained ``Statistics`` object on ``data``.

        Args:
            data: Iterable of elements passed to each chapter.

        Returns:
            Mapping of chapter name to that chapter's compiled record.
        """
        record = {}
        for name, stats in self.items():
            record[name] = stats.compile(data)
        return record

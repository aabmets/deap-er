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
from typing import Any

__all__ = ["init_repeat", "init_iterate", "init_cycle"]


def init_repeat(container: Callable[..., Any], func: Callable[..., Any], size: int) -> Any:
    """Call ``func`` ``size`` times and store the results in ``container``.

    Use with a Toolbox to register a generator of filled containers,
    such as individuals or a population.

    Args:
        container: Callable that takes an iterable and returns a collection.
        func: Function called once per element.
        size: Number of times to call ``func``.

    Returns:
        A collection filled with ``size`` results of ``func``.
    """
    return container(func() for _ in range(size))


def init_iterate(container: Callable[..., Any], generator: Callable[..., Any]) -> Any:
    """Call ``generator`` and store its results in ``container``.

    ``generator`` must return an iterable. Use with a Toolbox to
    register a generator of filled containers, as individuals or a
    population.

    Args:
        container: Callable that takes an iterable and returns a collection.
        generator: Function that returns the iterable used to fill the
            container.

    Returns:
        A collection filled with the results of ``generator``.
    """
    return container(generator())


def init_cycle(
    container: Callable[..., Any], funcs: Iterable[Callable[..., Any]], size: int = 1
) -> Any:
    """Call each function in ``funcs`` ``size`` times and store all results.

    Use with a Toolbox to register a generator of filled containers,
    as individuals or a population.

    Args:
        container: Callable that takes an iterable and returns a collection.
        funcs: Sequence of functions to call.
        size: Number of times to iterate through ``funcs``.

    Returns:
        A collection filled with the results of all function calls.
    """
    return container(func() for _ in range(size) for func in funcs)

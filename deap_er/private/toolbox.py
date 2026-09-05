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

from array import array
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy

if TYPE_CHECKING:
    from deap_er.private.typedefs import Individual

__all__ = ["clone_individual", "Toolbox"]


def clone_individual(individual: Individual) -> Individual:
    """Copy a sequence individual and its fitness without a full deepcopy.

    Register this on a Toolbox when a shallow gene copy is enough:
    ``toolbox.register("clone", clone_individual)``. Falls back to
    ``copy.deepcopy`` when the individual is a NumPy array or carries
    extra state such as ``strategy``, ``ps_``, or ``history_index``.

    Args:
        individual: Individual to copy.

    Returns:
        An independent copy of ``individual``.
    """
    extra = getattr(individual, "__dict__", None)
    if extra is not None and extra.keys() - {"fitness"}:
        return deepcopy(individual)
    if hasattr(individual, "strategy") or hasattr(individual, "ps_"):
        return deepcopy(individual)
    if hasattr(individual, "history_index"):
        return deepcopy(individual)
    if isinstance(individual, numpy.ndarray):
        return deepcopy(individual)
    if not isinstance(individual, list | array):
        return deepcopy(individual)

    clone = type(individual)(individual)
    if hasattr(individual, "fitness"):
        clone.fitness = deepcopy(individual.fitness)
    return clone


class Toolbox:
    """A container for evolutionary operators.

    Registers callables under aliases so algorithms can request
    ``mate``, ``mutate``, ``select``, ``evaluate``, and similar tools
    without hard-coding implementations.
    """

    def __init__(self) -> None:
        """Register the default ``clone`` and ``map`` operators."""
        self.register("clone", deepcopy)
        self.register("map", map)

    def __getattr__(self, name: str) -> Any:
        """Resolve aliases bound by ``register``.

        ``register`` attaches names with ``setattr``. This hook is
        for the type checker and for missing aliases at runtime.

        Args:
            name: Operator alias.

        Raises:
            AttributeError: If ``name`` is not registered.
        """
        raise AttributeError(name)

    def register(self, alias: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Bind ``func`` to ``alias`` on this toolbox.

        Extra positional and keyword arguments are bound into the
        registered callable. Callers may still override those bound
        values when they invoke the alias.

        Args:
            alias: Name to register. Overwrites an existing alias of the same name.
            func: Callable the alias will refer to.
            *args: Positional arguments bound into ``func``.
            **kwargs: Keyword arguments bound into ``func``.
        """
        p_func: Any = partial(func, *args, **kwargs)
        p_func.__name__ = alias
        p_func.__doc__ = func.__doc__

        if hasattr(func, "__dict__") and not isinstance(func, type):
            p_func.__dict__.update(func.__dict__.copy())
        setattr(self, alias, p_func)

    def unregister(self, alias: str) -> None:
        """Remove the operator registered as ``alias``.

        Args:
            alias: Name of the operator to remove.
        """
        delattr(self, alias)

    def decorate(self, alias: str, *decorators: Callable[..., Any]) -> None:
        """Wrap the operator ``alias`` with one or more decorators.

        Args:
            alias: Name of a registered operator.
            *decorators: Decorators applied left to right. If omitted, the
                operator is left unchanged.
        """
        if not decorators:
            return
        p_func = getattr(self, alias)
        func = p_func.func
        args = p_func.args
        kwargs = p_func.keywords
        for decorator in decorators:
            func = decorator(func)
        self.register(alias, func, *args, **kwargs)

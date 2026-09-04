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
from .lint_hints import LintHints
from typing import Any
from collections.abc import Callable
from functools import partial
from copy import deepcopy


__all__ = ["Toolbox"]


class Toolbox(LintHints):
    """A container for evolutionary operators.

    Registers callables under aliases so algorithms can request
    ``mate``, ``mutate``, ``select``, ``evaluate``, and similar tools
    without hard-coding implementations.
    """

    def __init__(self) -> None:
        """Register the default ``clone`` and ``map`` operators."""
        self.register("clone", deepcopy)
        self.register("map", map)

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

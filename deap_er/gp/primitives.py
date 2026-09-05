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

from collections.abc import Callable
from typing import Any, override

from ._primitive_nodes import Ephemeral, Primitive, Terminal
from ._primitive_set import PrimitiveSetTyped
from ._primitive_tree import PrimitiveTree

__all__ = [
    "Terminal",
    "Ephemeral",
    "Primitive",
    "PrimitiveTree",
    "PrimitiveSet",
    "PrimitiveSetTyped",
]


class PrimitiveSet(PrimitiveSetTyped):
    """Untyped primitive set.

    Subclass of ``PrimitiveSetTyped`` that treats every type as
    ``object``.

    Args:
        name: Name of the primitive set.
        arity: Number of input arguments.
        prefix: Prefix used to name input arguments.
    """

    def __init__(self, name: str, arity: int, prefix: str = "ARG") -> None:
        """Create an untyped set with ``arity`` inputs."""
        args: list[type] = [object] * arity
        super().__init__(name, args, object, prefix)

    @override
    def add_primitive(  # type: ignore[override]
        self, primitive: Callable[..., Any], arity: int, name: str | None = None, *_: Any, **__: Any
    ) -> None:
        """Add an untyped primitive of the given arity.

        Args:
            primitive: Callable to register.
            arity: Number of arguments. Must be at least 1.
            name: Optional name. Defaults to ``primitive.__name__``.

        Raises:
            ValueError: If ``arity`` is less than 1, or if ``name`` is
                already registered.
        """
        if arity < 1:
            raise ValueError("arity should be >= 1")
        args: list[type] = [object] * arity
        super().add_primitive(primitive, args, object, name)

    @override
    def add_terminal(  # type: ignore[override]
        self, terminal: Any, name: str | None = None, *_: Any, **__: Any
    ) -> None:
        """Add an untyped terminal to the set.

        Args:
            terminal: Value or callable to register as a terminal.
            name: Optional name. Defaults to ``terminal.__name__``
                when ``terminal`` is callable.
        """
        super().add_terminal(terminal, object, name)

    @override
    def add_ephemeral_constant(  # type: ignore[override]
        self, name: str, ephemeral: Callable[..., Any], *_: Any, **__: Any
    ) -> None:
        """Add an untyped ephemeral constant to the set.

        Args:
            name: Name of this ephemeral type.
            ephemeral: Zero-arity callable that produces a value.
        """
        super().add_ephemeral_constant(name, ephemeral, object)

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

import abc
from typing import Any, override

__all__: list[str] = ["Terminal", "Ephemeral", "Primitive"]


class Terminal:
    """Leaf node in a GP expression.

    A terminal is a value or a zero-arity function.

    Args:
        terminal: Value or zero-arity function stored in the leaf.
        symbolic: If True, format the value with ``str``; otherwise
            with ``repr``.
        ret_type: Return type of the terminal.
    """

    __slots__ = ("name", "value", "ret", "conv_fct", "call_zero")

    def __init__(
        self, terminal: Any, symbolic: bool, ret_type: type, call_zero: bool = False
    ) -> None:
        """Store ``terminal`` as a named leaf of ``ret_type``.

        Args:
            terminal: Value or name stored in the leaf.
            symbolic: If True, format the value with ``str``.
            ret_type: Return type of the terminal.
            call_zero: If True, format as a zero-arity call ``name()``.
        """
        self.ret = ret_type
        self.value = terminal
        self.name = str(terminal)
        self.conv_fct = str if symbolic else repr
        self.call_zero = call_zero

    @property
    def arity(self) -> int:
        """Number of arguments this terminal takes.

        Always 0.
        """
        return 0

    def format(self) -> str:
        """Return the string form of the terminal value."""
        text = self.conv_fct(self.value)
        if self.call_zero:
            return f"{text}()"
        return text

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is a terminal with the same slots."""
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot) for slot in self.__slots__)
        else:
            return NotImplemented


class Ephemeral(Terminal, abc.ABC):
    """Terminal whose value is sampled when the instance is created.

    Abstract base class. Subclasses must define a static method named
    ``func``.
    """

    def __init__(self) -> None:
        """Sample ``func`` and initialize as a non-symbolic terminal."""
        Terminal.__init__(self, self.func(), symbolic=False, ret_type=self.ret)

    @staticmethod
    @abc.abstractmethod
    def func() -> Any:
        """Produce a new ephemeral value.

        Subclasses must override this static method.

        Raises:
            NotImplementedError: If the subclass does not define ``func``.
        """
        raise NotImplementedError


class Primitive:
    """Function node in a GP expression.

    Formats as a Python call when given formatted child expressions.

    Args:
        name: Name of the primitive.
        args: Argument types of the primitive.
        ret_type: Return type of the primitive.
        weight: Relative sampling weight. Must be greater than 0.
    """

    __slots__ = ("name", "arity", "args", "ret", "seq", "weight")

    def __init__(self, name: str, args: list[type], ret_type: type, weight: float = 1.0) -> None:
        """Store the primitive name, argument types, and return type."""
        self.name = name
        self.arity = len(args)
        self.args = args
        self.ret = ret_type
        self.weight = weight
        placeholders = ", ".join(map("{{{0}}}".format, list(range(self.arity))))
        self.seq = f"{self.name}({placeholders})"

    def format(self, *args: str) -> str:
        """Format this primitive as a Python call.

        Args:
            *args: Formatted child expressions, one per argument.

        Returns:
            The primitive applied to ``args`` as source text.
        """
        return self.seq.format(*args)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is a primitive with the same slots."""
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot) for slot in self.__slots__)
        else:
            return NotImplemented

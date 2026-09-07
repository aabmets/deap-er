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

from collections import defaultdict
from collections.abc import Callable
from typing import Any, cast

from .primitive_nodes import Ephemeral, Primitive, Terminal

__all__: list[str] = ["PrimitiveSetTyped"]


class PrimitiveSetTyped:
    """Primitive set for strongly typed genetic programming.

    Args:
        name: Name of the primitive set.
        in_types: Input types, one per argument.
        ret_type: Return type of expressions built from this set.
        prefix: Prefix used to name input arguments.
    """

    def __init__(
        self, name: str, in_types: list[type], ret_type: type, prefix: str = "ARG"
    ) -> None:
        """Create an empty typed set and register one terminal per input."""
        self.name = name
        self.ins = in_types
        self.ret = ret_type

        self.terminals = defaultdict(list)
        self.primitives = defaultdict(list)
        self.context = {"__builtins__": None}
        self.arguments = []
        self.mapping = {}
        self.terms_count = 0
        self.prims_count = 0

        for i, type_ in enumerate(in_types):
            arg_str = f"{prefix}{i}"
            self.arguments.append(arg_str)
            term = Terminal(arg_str, True, type_)
            self._add_prim(term)
            self.terms_count += 1

    @staticmethod
    def _add_type(mapping: dict[Any, list[Any]], ret_type: Any) -> None:
        """Ensure ``mapping`` has a list for ``ret_type``.

        If the type is new, the list is filled with items already stored
        under compatible types.

        Args:
            mapping: Type-to-items dictionary to update.
            ret_type: Return type to register.
        """
        if ret_type not in mapping:
            new_list = []
            for type_, list_ in mapping.items():
                if issubclass(type_, ret_type):
                    for item in list_:
                        if item not in new_list:
                            new_list.append(item)
            mapping[ret_type] = new_list

    def _add_prim(self, prim: Primitive | Terminal | type[Ephemeral]) -> None:
        """Register ``prim`` in this set under its return type.

        Args:
            prim: Primitive, terminal, or ephemeral class to add.
        """
        self._add_type(self.primitives, prim.ret)
        self._add_type(self.terminals, prim.ret)
        self.mapping[prim.name] = prim

        if isinstance(prim, Primitive):
            for type_ in prim.args:
                self._add_type(self.primitives, type_)
                self._add_type(self.terminals, type_)
            mapping = self.primitives
        else:
            mapping = self.terminals

        for type_ in mapping:
            if not isinstance(type_, type):
                continue
            key = cast(type, type_)
            ret = cast(type, prim.ret)
            if issubclass(ret, key):
                mapping[key].append(prim)

    def add_primitive(
        self,
        primitive: Callable[..., Any],
        in_types: list[type],
        ret_type: type,
        name: str | None = None,
        weight: float = 1.0,
    ) -> None:
        """Add a primitive to the set.

        Args:
            primitive: Callable to register.
            in_types: Argument types of the primitive.
            ret_type: Type returned by the primitive.
            name: Optional name. Defaults to ``primitive.__name__``.
            weight: Relative sampling weight. Must be greater than 0.

        Raises:
            ValueError: If ``name`` is already registered, or if
                ``weight`` is not greater than 0.
        """
        if weight <= 0:
            raise ValueError("Primitive weight must be greater than 0.")
        if name is None:
            raw_name = getattr(primitive, "__name__", None)
            if not isinstance(raw_name, str):
                raise TypeError("Primitive must have a name or a '__name__' attribute.")
            name = raw_name

        if name in self.arguments:
            raise ValueError(
                f"Primitive name '{name}' is also an argument of the primitive set. "
                f"A compiled lambda parameter would shadow the primitive."
            )
        if name in self.context:
            raise ValueError(
                f"Primitives are required to have a unique name. "
                f"Consider using the argument 'name' to "
                f"rename your second '{name}' primitive."
            )
        prim = Primitive(name, in_types, ret_type, weight=weight)

        self._add_prim(prim)
        self.context[prim.name] = primitive
        self.prims_count += 1

    def add_terminal(
        self, terminal: Any, ret_type: type, name: str | None = None, *, call_zero: bool = False
    ) -> None:
        """Add a terminal to the set.

        Args:
            terminal: Value or callable to register as a terminal.
            ret_type: Type returned by the terminal.
            name: Optional name. Defaults to ``terminal.__name__``
                when ``terminal`` is callable.
            call_zero: If True, format a callable terminal as ``name()``
                so eval calls it. False keeps action terminals as names.

        Raises:
            ValueError: If ``name`` is already registered, or matches
                an argument of the primitive set.
        """
        symbolic = False
        if name is None and callable(terminal):
            raw_name = getattr(terminal, "__name__", None)
            name = raw_name if isinstance(raw_name, str) else None

        if name is not None and name in self.arguments:
            raise ValueError(
                f"Terminal name '{name}' is also an argument of the primitive set. "
                f"A compiled lambda parameter would shadow the terminal."
            )
        if name is not None and name in self.context:
            raise ValueError(
                f"Terminals are required to have a unique name. "
                f"Consider using the argument 'name' to "
                f"rename your second '{name}' terminal."
            )

        invoke_zero = False
        if name is not None:
            invoke_zero = call_zero and callable(terminal)
            self.context[name] = terminal
            terminal = name
            symbolic = True
        elif terminal in (True, False):
            self.context[str(terminal)] = terminal

        prim = Terminal(terminal, symbolic, ret_type, call_zero=invoke_zero)
        self._add_prim(prim)
        self.terms_count += 1

    def add_ephemeral_constant(
        self, name: str, ephemeral: Callable[..., Any], ret_type: type
    ) -> None:
        """Add an ephemeral constant to the set.

        An ephemeral is a zero-arity function that returns a random
        value. Each tree samples its own immutable value.

        Args:
            name: Name of this ephemeral type.
            ephemeral: Zero-arity callable that produces a value.
            ret_type: Type returned by the ephemeral.

        Raises:
            TypeError: If ``name`` is already used by a different
                ephemeral or by another class in this module.
        """
        module_gp = globals()
        if name not in module_gp:
            attrs = {"func": staticmethod(ephemeral), "ret": ret_type}
            class_ = type(name, (Ephemeral,), attrs)
            module_gp[name] = class_
        else:
            class_ = module_gp[name]
            if issubclass(class_, Ephemeral):
                if class_.func is not ephemeral:
                    raise TypeError(
                        "Ephemera with different functions should be "
                        "named differently even between psets."
                    )
                elif class_.ret is not ret_type:
                    raise TypeError(
                        "Ephemera with the same name and function should "
                        "have the same type even between psets."
                    )
            else:
                raise TypeError(
                    "Ephemera should be named differently than classes defined in the gp module."
                )

        self._add_prim(class_)
        self.terms_count += 1

    def add_adf(self, prim_set: PrimitiveSetTyped) -> None:
        """Add an Automatically Defined Function (ADF) to the set.

        Args:
            prim_set: Primitive set that defines the ADF name, inputs,
                and return type.
        """
        prim = Primitive(prim_set.name, prim_set.ins, prim_set.ret)
        self._add_prim(prim)
        self.prims_count += 1

    def rename_arguments(self, **kwargs: str) -> None:
        """Rename input arguments using the given mapping.

        Args:
            **kwargs: Map of current argument names to new names. Names
                that are not current arguments are ignored.
        """
        for i, old_name in enumerate(self.arguments):
            if old_name in kwargs:
                new_name = kwargs[old_name]
                self.arguments[i] = new_name
                self.mapping[new_name] = self.mapping[old_name]
                self.mapping[new_name].value = new_name
                del self.mapping[old_name]

    @property
    def terminal_ratio(self) -> float:
        """Ratio of terminals to all primitives in the set."""
        return self.terms_count / float(self.terms_count + self.prims_count)

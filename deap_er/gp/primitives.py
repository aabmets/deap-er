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
import ast
import copy
import re
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from typing import Any, cast, override

__all__ = [
    "Terminal",
    "Ephemeral",
    "Primitive",
    "PrimitiveTree",
    "PrimitiveSet",
    "PrimitiveSetTyped",
]


class Terminal:
    """Leaf node in a GP expression.

    A terminal is a value or a zero-arity function.

    Args:
        terminal: Value or zero-arity function stored in the leaf.
        symbolic: If True, format the value with ``str``; otherwise
            with ``repr``.
        ret_type: Return type of the terminal.
    """

    __slots__ = ("name", "value", "ret", "conv_fct")

    def __init__(self, terminal: Any, symbolic: bool, ret_type: type) -> None:
        """Store ``terminal`` as a named leaf of ``ret_type``."""
        self.ret = ret_type
        self.value = terminal
        self.name = str(terminal)
        self.conv_fct = str if symbolic else repr

    @property
    def arity(self) -> int:
        """Number of arguments this terminal takes.

        Always 0.
        """
        return 0

    def format(self) -> str:
        """Return the string form of the terminal value."""
        return self.conv_fct(self.value)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is a terminal with the same slots."""
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot) for slot in self.__slots__)
        else:
            return NotImplemented


class Ephemeral(Terminal):
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
    """

    __slots__ = ("name", "arity", "args", "ret", "seq")

    def __init__(self, name: str, args: list[type], ret_type: type) -> None:
        """Store the primitive name, argument types, and return type."""
        self.name = name
        self.arity = len(args)
        self.args = args
        self.ret = ret_type
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
    ) -> None:
        """Add a primitive to the set.

        Args:
            primitive: Callable to register.
            in_types: Argument types of the primitive.
            ret_type: Type returned by the primitive.
            name: Optional name. Defaults to ``primitive.__name__``.

        Raises:
            ValueError: If ``name`` is already registered in the set.
        """
        if name is None:
            raw_name = getattr(primitive, "__name__", None)
            if not isinstance(raw_name, str):
                raise TypeError("Primitive must have a name or a '__name__' attribute.")
            name = raw_name

        if name in self.context:
            raise ValueError(
                f"Primitives are required to have a unique name. "
                f"Consider using the argument 'name' to "
                f"rename your second '{name}' primitive."
            )
        prim = Primitive(name, in_types, ret_type)

        self._add_prim(prim)
        self.context[prim.name] = primitive
        self.prims_count += 1

    def add_terminal(self, terminal: Any, ret_type: type, name: str | None = None) -> None:
        """Add a terminal to the set.

        Args:
            terminal: Value or callable to register as a terminal.
            ret_type: Type returned by the terminal.
            name: Optional name. Defaults to ``terminal.__name__``
                when ``terminal`` is callable.

        Raises:
            ValueError: If ``name`` is already registered in the set.
        """
        symbolic = False
        if name is None and callable(terminal):
            raw_name = getattr(terminal, "__name__", None)
            name = raw_name if isinstance(raw_name, str) else None

        if name is not None and name in self.context:
            raise ValueError(
                f"Terminals are required to have a unique name. "
                f"Consider using the argument 'name' to "
                f"rename your second '{name}' terminal."
            )

        if name is not None:
            self.context[name] = terminal
            terminal = name
            symbolic = True
        elif terminal in (True, False):
            self.context[str(terminal)] = terminal

        prim = Terminal(terminal, symbolic, ret_type)
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


class PrimitiveTree(list[Any]):
    """Prefix-ordered tree of primitives and terminals.

    A list subclass used by genetic programming operators. Every node
    must expose an ``arity`` attribute.

    Args:
        content: Primitives and terminals that form the tree.
    """

    def __init__(self, content: Iterable[Any]) -> None:
        """Initialize the tree from ``content``."""
        super().__init__(content)

    def __deepcopy__(self, memo: dict[int, Any]) -> PrimitiveTree:
        """Return a deep copy of this tree.

        Args:
            memo: Memo mapping used by ``copy.deepcopy``.

        Returns:
            A new tree with copied contents and attributes.
        """
        new = self.__class__(self)
        new.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return new

    @override
    def __setitem__(self, key: Any, val: Any) -> None:  # type: ignore[override]
        """Replace a node or subtree, preserving tree arity.

        Args:
            key: Index or slice of the node or subtree to replace.
            val: Replacement node or sequence of nodes.

        Raises:
            IndexError: If a slice starts past the end of the tree.
            ValueError: If the replacement would change the tree arity.
        """
        if isinstance(key, slice):
            if key.start >= len(self):
                raise IndexError(
                    "Trying to set a slice larger than the size "
                    "of the PrimitiveTree is not allowed."
                )
            total = val[0].arity
            for node in val[1:]:
                total += node.arity - 1
            if total != 0:
                raise ValueError(
                    "Insertion of a subtree with an arity smaller "
                    "than the PrimitiveTree is not allowed."
                )
        elif val.arity != self[key].arity:
            raise ValueError(
                "PrimitiveTree node replacement with a node of a different arity is not allowed."
            )
        list.__setitem__(self, key, val)

    @override
    def __str__(self) -> str:
        """Return the tree as a Python expression string."""
        string = ""
        stack: list[Any] = []
        for node in self:
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()
                string = prim.format(*args)
                if len(stack) == 0:
                    break
                stack[-1][1].append(string)
        return str(string)

    @classmethod
    def from_string(cls, string: str, prim_set: PrimitiveSetTyped) -> PrimitiveTree:
        """Build a tree from a Python expression string.

        ``prim_set`` must contain every primitive that appears in
        ``string``.

        Args:
            string: Python expression to deserialize.
            prim_set: Primitive set used to resolve names.

        Returns:
            A tree populated with the deserialized primitives.

        Raises:
            TypeError: If a token is not a registered primitive
                and is not a Python literal, or if a primitive or
                terminal type does not match the expected type.
        """
        tokens = re.split("[ \t\n\r\f\v(),]", string)
        expr = []
        ret_types = deque()
        for token in tokens:
            if token == "":
                continue
            ret_type = ret_types.popleft() if len(ret_types) != 0 else None

            if token in prim_set.mapping:
                primitive = prim_set.mapping[token]
                if ret_type is not None and not issubclass(primitive.ret, ret_type):
                    raise TypeError(
                        f"Primitive {primitive} return type {primitive.ret} "
                        f"does not match the expected one: {ret_type}."
                    )
                expr.append(primitive)
                if isinstance(primitive, Primitive):
                    ret_types.extendleft(reversed(primitive.args))
            else:
                try:
                    token = ast.literal_eval(token)
                except (ValueError, SyntaxError) as err:
                    raise TypeError(f"Unable to evaluate terminal: {token}.") from err
                if ret_type is None:
                    ret_type = type(token)
                if not issubclass(type(token), ret_type):
                    raise TypeError(
                        f"Terminal {token} type {type(token)} does "
                        f"not match the expected one: {ret_type}."
                    )
                prim = Terminal(token, False, ret_type)
                expr.append(prim)
        return cls(expr)

    def search_subtree(self, begin: int) -> slice:
        """Return the slice of the subtree rooted at ``begin``.

        Args:
            begin: Index of the subtree root.

        Returns:
            Slice covering that subtree.
        """
        end = begin + 1
        total = self[begin].arity
        while total > 0:
            total += self[end].arity - 1
            end += 1
        return slice(begin, end)

    @property
    def height(self) -> int:
        """Height of the tree, which is the depth of the deepest node."""
        stack = [0]
        max_depth = 0
        for elem in self:
            depth = stack.pop()
            max_depth = max(max_depth, depth)
            stack.extend([depth + 1] * elem.arity)
        return int(max_depth)

    @property
    def root(self) -> Any:
        """Root node of the tree (the first element)."""
        return self[0]

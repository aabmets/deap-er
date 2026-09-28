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
import array
import warnings
from typing import Any, cast

from .overrides import ArrayOverride, NumpyOverride

__all__: list[str] = ["create_type"]

type _Definition = tuple[Any, str | None, dict[str, Any], dict[str, Any]]

_definitions: dict[str, tuple[_Definition, type]] = {}


def create_type(name: str, base: type | object, **kwargs: Any) -> type:
    """Create a class named ``name`` and register it on the ``creator`` module.

    The new class inherits from ``base``. Each keyword argument becomes
    an attribute: a *class object* is stored as an instance attribute
    (instantiated when each individual is created); any other value is
    stored as a class attribute.

    Calling it again with the same name and an equal definition returns
    the existing class unchanged, so live instances and pickles keep
    working. A different definition under an existing name warns and
    replaces the old class.

    Args:
        name: Name of the class to create.
        base: Type or instance to inherit from. An instance is replaced
            by its class.
        **kwargs: Attributes added to the new class.

    Returns:
        The created class, or the existing one for an equal definition.

    Raises:
        ValueError: If ``name`` is one of the ``creator`` module's own
            attributes, such as ``create_type`` or ``array``.
    """
    if name in _RESERVED:
        raise ValueError(f"'{name}' is reserved by the creator module; choose another name.")

    array_typecode = None
    if type(base) is array.array:
        array_typecode = base.typecode
        base = type(base)

    # set base to class if base is an instance
    if not hasattr(base, "__module__"):
        base = base.__class__

    # override numpy and array classes
    base = {"array": ArrayOverride, "numpy": NumpyOverride}.get(base.__module__, base)

    # separate kwargs by their type
    inst_attr, cls_attr = {}, {}
    for key, value in kwargs.items():
        condition = type(value) is type
        _dict = inst_attr if condition else cls_attr
        _dict[key] = value
    if array_typecode is not None:
        cls_attr.setdefault("typecode", array_typecode)

    definition: _Definition = (base, array_typecode, inst_attr, cls_attr)
    existing = globals().get(name)
    known = _definitions.get(name)
    if existing is not None:
        if known is not None and known[1] is existing and _same_definition(known[0], definition):
            return existing
        msg = (
            f"You are creating a new class named '{name}', "
            f"which already exists. The old definition will "
            f"be overwritten by the new one."
        )
        warnings.warn(stacklevel=2, message=msg, category=RuntimeWarning)

    # create the new class
    new_class = type(name, (cast(Any, base),), cls_attr)

    # define the replacement init func
    def new_init_func(self, *args_: Any, **kwargs_: Any) -> None:
        for attr_name, attr_obj in inst_attr.items():
            setattr(self, attr_name, attr_obj())
        if base.__init__ is not object.__init__:
            cast(Any, base.__init__)(self, *args_, **kwargs_)

    # override the init func and set the global name
    new_class.__init__ = new_init_func
    globals()[name] = new_class
    _definitions[name] = (definition, new_class)
    # The base is only known at runtime, so ty cannot see the class it makes.
    return new_class  # ty: ignore[unsound-return-statement]


def _same_definition(left: _Definition, right: _Definition) -> bool:
    """Compare bases and classes by identity and other values by ``==``."""
    if left[0] is not right[0] or left[1] != right[1]:
        return False
    return all(_same_attrs(a, b) for a, b in zip(left[2:], right[2:], strict=True))


def _same_attrs(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.keys() != right.keys():
        return False
    return all(_same_value(left[key], right[key]) for key in left)


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, type) or isinstance(right, type):
        return left is right
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


_RESERVED = frozenset(globals()) | {"_RESERVED"}

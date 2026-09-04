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
import warnings
from typing import Any, cast

from .overrides import _ArrayOverride, _NumpyOverride

__all__ = ["create"]


def create(name: str, base: type | object, **kwargs: Any) -> None:
    """Create a class named ``name`` and register it on the ``creator`` module.

    The new class inherits from ``base``. Each keyword argument becomes
    an attribute: a *class object* is stored as an instance attribute
    (instantiated when each individual is created); any other value is
    stored as a class attribute.

    Warns if ``name`` already exists on the module; the old definition
    is overwritten.

    Args:
        name: Name of the class to create.
        base: Type or instance to inherit from. An instance is replaced
            by its class.
        **kwargs: Attributes added to the new class.
    """
    # warn about class definition overwrite
    if name in globals():
        msg = (
            f"You are creating a new class named '{name}', "
            f"which already exists. The old definition will "
            f"be overwritten by the new one."
        )
        warnings.warn(stacklevel=2, message=msg, category=RuntimeWarning)

    # set base to class if base is an instance
    if not hasattr(base, "__module__"):
        base = base.__class__

    # override numpy and array classes
    base = {"array": _ArrayOverride, "numpy": _NumpyOverride}.get(base.__module__, base)

    # separate kwargs by their type
    inst_attr, cls_attr = {}, {}
    for key, value in kwargs.items():
        condition = type(value) is type
        _dict = inst_attr if condition else cls_attr
        _dict[key] = value

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

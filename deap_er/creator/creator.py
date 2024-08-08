#
#   MIT License
#   
#   Copyright (c) 2022, Mattias Aabmets
#   
#   The contents of this file are subject to the terms and conditions defined in the License.
#   You may not use, modify, or distribute this file except in compliance with the License.
#   
#   SPDX-License-Identifier: MIT
#
from .overrides import *
from typing import Optional, Union
import warnings


__all__ = ['create']


# ====================================================================================== #
def create(name: str, base: Union[type, object], **kwargs: Optional) -> None:
    """
    Creates a new class named **name**, which inherits from the **base** class, and
    registers it into the global namespace of the *creator* module. Any optional
    **kwargs** provided to this function will be set as attributes of the new class.

    :param name: The name of the new class to create.
    :param base: A type or an object from which to inherit.
    :param kwargs: One or more keyword arguments to add to the new class
        as attributes, optional. If a kwarg is an instance, it will
        be added as a class attribute. If a kwarg is a class, it
        will be instantiated and added as an instance attribute.
    :return: Nothing.
    """
    # warn about class definition overwrite
    if name in globals():
        msg = f"You are creating a new class named \'{name}\', " \
              f"which already exists. The old definition will " \
              f"be overwritten by the new one."
        warnings.warn(
            message=msg,
            category=RuntimeWarning
        )

    # set base to class if base is an instance
    if not hasattr(base, '__module__'):
        base = base.__class__

    # override numpy and array classes
    base = dict(
        array=_ArrayOverride,
        numpy=_NumpyOverride
    ).get(base.__module__, base)

    # separate kwargs by their type
    inst_attr, cls_attr = dict(), dict()
    for key, value in kwargs.items():
        condition = type(value) is type
        _dict = inst_attr if condition else cls_attr
        _dict[key] = value

    # create the new class
    new_class = type(name, tuple([base]), cls_attr)

    # define the replacement init func
    def new_init_func(self, *args_, **kwargs_):
        for attr_name, attr_obj in inst_attr.items():
            setattr(self, attr_name, attr_obj())
        if base.__init__ is not object.__init__:
            base.__init__(self, *args_, **kwargs_)

    # override the init func and set the global name
    new_class.__init__ = new_init_func
    globals()[name] = new_class

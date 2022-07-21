# ====================================================================================== #
#                                                                                        #
#   MIT License                                                                          #
#                                                                                        #
#   Copyright (c) 2022 - Mattias Aabmets, The DEAP Team and Other Contributors           #
#                                                                                        #
#   Permission is hereby granted, free of charge, to any person obtaining a copy         #
#   of this software and associated documentation files (the "Software"), to deal        #
#   in the Software without restriction, including without limitation the rights         #
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell            #
#   copies of the Software, and to permit persons to whom the Software is                #
#   furnished to do so, subject to the following conditions:                             #
#                                                                                        #
#   The above copyright notice and this permission notice shall be included in all       #
#   copies or substantial portions of the Software.                                      #
#                                                                                        #
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR           #
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,             #
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE          #
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER               #
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,        #
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE        #
#   SOFTWARE.                                                                            #
#                                                                                        #
# ====================================================================================== #
from .overrides import *
from typing import Optional, Union
import warnings


__all__ = ['create']


# ====================================================================================== #
class _DevTestClass:  # pragma: no cover
    def __init__(self, *_, **__):
        pass


# ====================================================================================== #
def create(name: str, base: Union[type, object], **kwargs: Optional) -> None:
    """
    Creates a new class named **name**, which inherits from the **base** class,
    and registers it into the global namespace. Any optional **kwargs** provided
    to this function will be set as attributes of the new class.

    Parameters:
        name: The name of the new class to create.
        base: A type or an object from which to inherit.
        kwargs: One or more keyword arguments to add to the new class
            as attributes, optional. If a kwarg is an instance, it will
            be added as a class attribute. If a kwarg is a class, it
            will be instantiated and added as an instance attribute.
    Returns:
        None
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
    def new_init_func(self, *_, **__):
        for attr_name, attr_obj in inst_attr.items():
            setattr(self, attr_name, attr_obj())

    # override the init func and set the global name
    new_class.__init__ = new_init_func
    globals()[name] = new_class

# ====================================================================================== #
#                                                                                        #
#   MIT License                                                                          #
#                                                                                        #
#   Copyright (c) 2022 - Mattias Aabmets and Contributors                                #
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
from .overrides import NumpyOverride, ArrayOverride
import warnings


# ====================================================================================== #
def create(name: str, base: type | object, **kwargs) -> None:
    """
    Creates a new class named *name*, which inherits from the *base* class.
    Any optional kwargs provided to this function will be set as attributes of the new class.
    If a kwarg is a class, an instance of it is added to an instance of the new class upon instantiation.
    Otherwise, if a kwarg is an instance, it is directly added to the new class as an attribute.

    :param name: The name of the new class to create.
    :param base: A base class from which to inherit.
    :param kwargs: One or more keyword arguments to add to the
        new class as class or instance attributes (optional).
    """

    warning = f"You are creating a new class named \'{name}\', which already exists. " \
              f"The old definition will be overwritten by the new one."

    # warn about class definition overwrite
    if name in globals():
        warnings.warn(
            message=warning,
            category=RuntimeWarning
        )

    # separate kwargs by their type
    classes, instances = dict(), dict()
    for key, value in kwargs.items():
        _dict = classes if type(value) is type else instances
        _dict[key] = value

    # override array and numpy classes
    if hasattr(base, '__module__'):
        base = dict(
            array=ArrayOverride,
            numpy=NumpyOverride
        ).get(base.__module__, base)

    # set base to class if base is an instance
    if not isinstance(base, type):
        base = base.__class__

    # define the replacement init func
    def new_init_func(self, *args, **kwargs_):
        for class_name, class_object in classes.items():
            setattr(self, class_name, class_object())
        if base.__init__ is not object.__init__:
            base.__init__(self, *args, **kwargs_)

    # create and register the new class
    new_class = type(name, (base,), instances)
    new_class.__init__ = new_init_func
    globals()[name] = new_class

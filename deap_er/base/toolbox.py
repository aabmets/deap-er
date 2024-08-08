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
from .lint_hints import LintHints
from typing import Callable, Optional
from functools import partial
from copy import deepcopy


__all__ = ['Toolbox']


# ====================================================================================== #
class Toolbox(LintHints):
    """
    A container for evolutionary operators. Toolboxes are essential
    components which facilitate the process of computational evolution.
    """
    # -------------------------------------------------------- #
    def __init__(self):
        self.register("clone", deepcopy)
        self.register("map", map)

    # -------------------------------------------------------- #
    def register(self, alias: str, func: Callable,
                 *args: Optional, **kwargs: Optional) -> None:
        """
        Registers a **func** in the toolbox under the name **alias**.
        Any **args** or **kwargs** will be automatically passed to the
        registered function when it's called. Fixed arguments can
        be overridden at function call time.

        :param alias: The name to register the 'func' under.
            The alias will be overwritten if it already exists.
        :param func: The function to which the alias is going to refer.
        :param args: Positional arguments which are automatically
            passed to the 'func' when it's called, optional.
        :param kwargs: Keyword arguments which are automatically
            passed to the 'func' when it's called, optional.
        :return: Nothing.
        """
        p_func = partial(func, *args, **kwargs)
        p_func.__name__ = alias
        p_func.__doc__ = func.__doc__

        if hasattr(func, '__dict__') and not isinstance(func, type):
            p_func.__dict__.update(func.__dict__.copy())
        setattr(self, alias, p_func)

    # -------------------------------------------------------- #
    def unregister(self, alias: str) -> None:
        """
        Removes an operator with the name **alias** from the toolbox.

        :param alias: The name of the operator to remove from the toolbox.
        :return: Nothing.
        """
        delattr(self, alias)

    # -------------------------------------------------------- #
    def decorate(self, alias: str,
                 *decorators: Optional[Callable]) -> None:
        """
        Decorates an operator **alias** with the provided **decorators**.

        :param alias: Name of the operator to decorate. The 'alias'
            must be a registered operator in the toolbox.
        :param decorators: Positional arguments of decorator functions
            to apply to the 'alias', optional. If none are provided,
            the operator is left unchanged. If multiple are provided,
            they are applied in order of iteration over the 'decorators'.
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

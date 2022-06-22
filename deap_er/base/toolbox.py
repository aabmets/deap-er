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
from typing import Callable
from functools import partial
from copy import deepcopy


__all__ = ['Toolbox']


# ====================================================================================== #
class LintHints:

    __test__: Callable

    generate: Callable
    evaluate: Callable
    update: Callable
    mutate: Callable
    select: Callable
    mate: Callable


# ====================================================================================== #
class Toolbox(LintHints):
    """
    A container for evolutionary operators.
    `Reference <https://deap.readthedocs.io/en/master/api/base.html#toolbox>`_
    """

    # -------------------------------------------------------------------------------------- #
    clone: partial
    map: partial

    # -------------------------------------------------------------------------------------- #
    def __init__(self):
        self.register("clone", deepcopy)
        self.register("map", map)

    # -------------------------------------------------------------------------------------- #
    def register(self, alias: str, func: Callable, *args, **kwargs) -> None:
        """
        Sets the *func* param with the name *alias* as an attribute to
        the *Toolbox* instance this method is called on.

        :param alias: Name of the operator to make the func available from.
            If a toolbox already has an operator with the same name, it will be overwritten.
        :param func: The function to which the alias is going to refer.
        :param args: Positional arguments which get passed to the func when it's called.
        :param kwargs: Keyword arguments which get passed to the func when it's called.
        :return: None
        """
        p_func = partial(func, *args, **kwargs)
        p_func.__name__ = alias
        p_func.__doc__ = func.__doc__

        if hasattr(func, '__dict__') and not isinstance(func, type):
            p_func.__dict__.update(func.__dict__.copy())
        setattr(self, alias, p_func)

    # -------------------------------------------------------------------------------------- #
    def unregister(self, alias) -> None:
        """
        Removes an operator with the name *alias* from the toolbox.

        :param alias: Name of the alias to remove from the toolbox.
        :return: None
        """
        delattr(self, alias)

    # -------------------------------------------------------------------------------------- #
    def decorate(self, alias: str, *decorators: Callable) -> None:
        """
        Decorates an operator *alias* with the provided *decorators*,
        where *alias* must be a registered operator in the toolbox.

        :param alias: Name of the operator to decorate.
        :param decorators: A list of decorator functions to apply to the alias.
        :return: None
        """
        p_func = getattr(self, alias)
        func = p_func.func
        args = p_func.args
        kwargs = p_func.keywords
        for decorator in decorators:
            func = decorator(func)
        self.register(alias, func, *args, **kwargs)

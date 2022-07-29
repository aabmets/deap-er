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
from collections.abc import Callable, Iterable


__all__ = ['init_repeat', 'init_iterate', 'init_cycle']


# ====================================================================================== #
def init_repeat(container: Callable, func: Callable, size: int) -> Iterable:
    """
    Calls the **func** argument **count** times and puts the results
    into an instance of **container**. This helper function can be used
    in conjunction with a Toolbox to register a generator of filled
    containers, such as individuals or a population.

    :param container: A callable which takes an iterable as argument
        and returns a :external+python:py:class:`~collections.abc.Collection`.
    :param func: The function to be called count times.
    :param size: The number of times to call the func.
    :return: An iterable filled with count results of func.
    """
    return container(func() for _ in range(size))


# -------------------------------------------------------------------------------------- #
def init_iterate(container: Callable, generator: Callable) -> Iterable:
    """
    Calls the **generator** function and puts the results into an instance
    of **container**. The **generator** function should return an iterable.
    This helper function can be used in conjunction with a Toolbox to register
    a generator of filled containers, as individuals or a population.

    :param container: A callable which takes an iterable as argument
        and returns a :external+python:py:class:`~collections.abc.Collection`.
    :param generator: A function returning an iterable to fill the container with.
    :return: An iterable filled with the results of the generator.
    """
    return container(generator())


# -------------------------------------------------------------------------------------- #
def init_cycle(container: Callable, funcs: Iterable, size: int = 1) -> Iterable:
    """
    Calls each function in the **funcs** iterable **count** times and stores
    the results from all function calls into the **container**. This helper
    function can be used in conjunction with a Toolbox to register a generator
    of filled containers, as individuals or a population.

    :param container: A callable which takes an iterable as argument
        and returns a :external+python:py:class:`~collections.abc.Collection`.
    :param funcs: A sequence of functions to be called.
    :param size: Number of times to iterate through the sequence of functions.
    :return: An iterable filled with the results of all function calls.
    """
    return container(func() for _ in range(size) for func in funcs)

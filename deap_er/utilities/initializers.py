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

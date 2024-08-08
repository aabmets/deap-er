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
from typing import Callable, Optional, Iterable
from functools import partial


__all__ = ['Statistics', 'MultiStatistics']


# ====================================================================================== #
class Statistics:
    """
    Object that compiles statistics on a list of arbitrary objects.
    When created, the statistics object receives a **key** argument that
    is used to get the values on which the statistics will be computed.
    If not provided, the **key** argument defaults to the identity function.

    The value returned by the key may be a multidimensional object, i.e.:
    a tuple or a list, as long as the registered statistical function
    supports it. For example, statistics can be computed directly on
    multi-objective fitness when using numpy statistical function.

    :param key: A function that takes an object and returns a
        value on which the statistics will be computed.
    """
    # -------------------------------------------------------- #
    def __init__(self, key: Optional[Callable] = None):
        self.key = key if key else lambda obj: obj
        self.functions = dict()
        self.fields = list()

    # -------------------------------------------------------- #
    def register(self, name: str, func: Callable,
                 *args: Optional, **kwargs: Optional) -> None:
        """
        Registers a new statistical function that will be applied
        to the sequence each time the *record* method is called.

        :param name: The name of the statistics function as it would
            appear in the dictionary of the statistics object.
        :param func: A function that will compute the desired
            statistics on the data as preprocessed by the key.
        :param args: Positional arguments to be passed to the function, optional.
        :param kwargs: Keyword arguments to be passed to the function, optional.
        :return: Nothing.
        """
        self.functions[name] = partial(func, *args, **kwargs)
        self.fields.append(name)

    # -------------------------------------------------------- #
    def compile(self, data: Iterable) -> dict:
        """
        Compiles the statistics on the given data.

        :param data: The data on which the statistics will be computed.
        :return: A dictionary containing the statistics.
        """
        entry = dict()
        values = tuple(self.key(elem) for elem in data)
        for key, func in self.functions.items():
            entry[key] = func(values)
        return entry


# ====================================================================================== #
class MultiStatistics(dict):
    """
    Object that compiles statistics on a list of arbitrary objects.
    Allows computation of statistics on multiple keys using a single
    call to the 'compile' method.
    """
    # -------------------------------------------------------- #
    @property
    def fields(self):
        return sorted(self.keys())

    # -------------------------------------------------------- #
    def register(self, name: str, func: Callable,
                 *args: Optional, **kwargs: Optional) -> None:
        """
        Registers a new statistical function that will be applied
        to the sequence each time the *record* method is called.

        :param name: The name of the statistics function as it would
            appear in the dictionary of the statistics object.
        :param func: A function that will compute the desired
            statistics on the data as preprocessed by the key.
        :param args: Positional arguments to be passed to the function, optional.
        :param kwargs: Keyword arguments to be passed to the function, optional.
        :return: Nothing.
        """
        for stats in self.values():
            stats.register(name, func, *args, **kwargs)

    # -------------------------------------------------------- #
    def compile(self, data: Iterable) -> dict:
        """
        Compiles the statistics on the given data.

        :param data: The data on which the statistics will be computed.
        :return: A dictionary containing the statistics.
        """
        record = dict()
        for name, stats in self.items():
            record[name] = stats.compile(data)
        return record

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
from functools import partial
from typing import Callable, Sequence


__all__ = ['Statistics', 'MultiStatistics']


# ====================================================================================== #
class Statistics:
    """
    Object that compiles statistics on a list of arbitrary objects.
    When created, the statistics object receives a *key* argument that
    is used to get the values on which the statistics will be computed.
    If not provided, the *key* argument defaults to the identity function.

    The value returned by the key may be a multidimensional object, i.e.:
    a tuple or a list, as long as the registered statistical function
    supports it. For example, statistics can be computed directly on
    multi-objective fitness when using numpy statistical function.
    """
    # -------------------------------------------------------------------------------------- #
    def __init__(self, key: Callable = None):
        self.key = key if key else lambda obj: obj
        self.functions = dict()
        self.fields = list()

    # -------------------------------------------------------------------------------------- #
    def register(self, name: str, func: Callable, *args, **kwargs) -> None:
        """
        Registers a new statistical function that will be applied on
        the sequence each time the *record* method is called.

        :param name: The name of the statistics function as it would
            appear in the dictionary of the statistics object.
        :param func: A function that will compute the desired
            statistics on the data as preprocessed by the key.
        :param args: Positional arguments to be passed to the function, optional.
        :param kwargs: Keyword arguments to be passed to the function, optional.
        :returns: None
        """
        self.functions[name] = partial(func, *args, **kwargs)
        self.fields.append(name)

    # -------------------------------------------------------------------------------------- #
    def compile(self, data: Sequence) -> dict:
        """
        Compiles the statistics on the given data.

        :param data: The data on which the statistics will be computed.
        :returns: A dictionary of the computed statistics.
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
    call to the *compile* method.
    """
    # -------------------------------------------------------------------------------------- #
    @property
    def fields(self):
        return sorted(self.keys())

    # -------------------------------------------------------------------------------------- #
    def register(self, name: str, func: Callable, *args, **kwargs) -> None:
        """
        Registers a new statistical function that will be applied on
        the sequence each time the *record* method is called.

        :param name: The name of the statistics function as it would
            appear in the dictionary of the statistics object.
        :param func: A function that will compute the desired
            statistics on the data as preprocessed by the key.
        :param args: Positional arguments to be passed to the function, optional.
        :param kwargs: Keyword arguments to be passed to the function, optional.
        :returns: None
        """
        for stats in self.values():
            stats.register(name, func, *args, **kwargs)

    # -------------------------------------------------------------------------------------- #
    def compile(self, data: Sequence) -> dict:
        """
        Compiles the statistics on the given data.

        :param data: The data on which the statistics will be computed.
        :returns: A dictionary of the computed statistics.
        """
        record = dict()
        for name, stats in self.items():
            record[name] = stats.compile(data)
        return record

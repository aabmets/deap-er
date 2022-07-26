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
from typing import Callable, Union, Iterable
from itertools import repeat
from functools import wraps
import numpy


# ====================================================================================== #
class Translate:
    """
    A decorator for evaluation functions. When the decorated function is called, the
    input :ref:`Individual <datatypes>` is translated by the preset list of **vector**
    values, the result of which is then passed into the evaluation function as a plain
    list of values. This decorator adds the :func:`translate` method to the decorated
    function, which can be used to update the translation vector values.

    :param vector: The translation vector to be used.
        Must have the same length as the individual.
    """
    vector = None

    # -------------------------------------------------------- #
    def __init__(self, vector: list):
        self.translate(vector)

    # -------------------------------------------------------- #
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(individual, *args, **kwargs):
            mod_ind = [v - t for v, t in zip(individual, self.vector)]
            return func(mod_ind, *args, **kwargs)
        wrapper.translate = self.translate
        return wrapper

    # -------------------------------------------------------- #
    def translate(self, vector: list) -> None:
        """
        Updates the translation **vector** values.
        After decorating the evaluation function, this method
        is directly available from the function object, which
        can be used to update the translation vector values.

        :param vector: The translation vector.
        """
        self.vector = vector


# -------------------------------------------------------------------------------------- #
class Rotate:
    """
    A decorator for evaluation functions. When the decorated function is called, the
    input :ref:`Individual <datatypes>` is rotated by the preset ndarray of **matrix**
    values, the result of which is then passed into the evaluation function as a plain
    ndarray of values. This decorator adds a :func:`rotate` method to the decorated
    function, which can be used to update the rotation matrix values.

    :param matrix: The rotation matrix to be used. Must be a valid orthogonal
        N * N rotation matrix, where N is the length of the individual.
    """
    matrix = None

    # -------------------------------------------------------- #
    def __init__(self, matrix):
        self.rotate(matrix)

    # -------------------------------------------------------- #
    def __call__(self, func):
        @wraps(func)
        def wrapper(individual, *args, **kwargs):
            mod_ind = numpy.dot(self.matrix, individual)
            return func(mod_ind, *args, **kwargs)
        wrapper.rotate = self.rotate
        return wrapper

    # -------------------------------------------------------- #
    def rotate(self, matrix):
        """
        Updates the rotation **matrix** values.
        After decorating the evaluation function, this method
        is directly available from the function object, which
        can be used to update the rotation matrix values.

        :param matrix: The rotation matrix.
        """
        self.matrix = numpy.linalg.inv(matrix)


# -------------------------------------------------------------------------------------- #
class Scale:
    """
    A decorator for evaluation functions. When the decorated function is called, the
    input :ref:`Individual <datatypes>` is scaled by the preset list of **factor**
    values, the result of which is then passed into the evaluation function as a plain
    list of values. This decorator adds the :func:`scale` method to the decorated
    function, which can be used to update the scale factor values.

    :param factor: The scale factor to be used.
        Must have the same length as the individual.
    """
    factor = None

    # -------------------------------------------------------- #
    def __init__(self, factor):
        self.scale(factor)

    # -------------------------------------------------------- #
    def __call__(self, func):
        @wraps(func)
        def wrapper(individual, *args, **kwargs):
            mod_ind = [v * f for v, f in zip(individual, self.factor)]
            return func(mod_ind, *args, **kwargs)
        wrapper.scale = self.scale
        return wrapper

    # -------------------------------------------------------- #
    def scale(self, factor: list) -> None:
        """
        Updates the scale **factor** values.
        After decorating the evaluation function, this method
        is directly available from the function object, which
        can be used to update the scale factor values.

        :param factor: The scale factor.
        """
        self.factor = tuple(1.0 / f for f in factor)


# -------------------------------------------------------------------------------------- #
class Noise:
    """
    A decorator for evaluation functions. When the decorated function is called,
    noise is added to the result(s) of the decorated function by the preset callable(s),
    which are called without arguments. This decorator adds the :func:`noise` method to
    the decorated functions, which can be used to update the noise generator callables.

    :param funcs: The noise generator callables.
        Can either be a single callable, which is applied to all values
        in the results object, or a list of callables, which must have the
        same length as the input individual. The values in the funcs argument
        can also be of type :obj:`None`, which prevents noise from being
        added to the corresponding value(s).
    """
    rand_funcs = None

    # -------------------------------------------------------- #
    def __init__(self, funcs):
        self.noise(funcs)

    # -------------------------------------------------------- #
    def __call__(self, func):
        @wraps(func)
        def wrapper(individual, *args, **kwargs):
            result = func(individual, *args, **kwargs)
            if not isinstance(result, Iterable):
                result = (result,)
            noisy = list()
            for r, f in zip(result, self.rand_funcs):
                if f is None:
                    noisy.append(r)
                else:
                    noisy.append(r + f())
            return tuple(noisy)
        wrapper.noise = self.noise
        return wrapper

    # -------------------------------------------------------- #
    def noise(self, funcs: Union[Callable, list[Callable]]) -> None:
        """
        Updates the noise generator **funcs**.
        After decorating the evaluation function, this method
        is directly available from the function object.

        :param funcs: The noise function(s).
        """
        if not isinstance(funcs, Iterable):
            self.rand_funcs = repeat(funcs)

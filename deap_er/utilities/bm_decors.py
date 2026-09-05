#
#   Apache License 2.0
#
#   Copyright (c) 2022, Mattias Aabmets
#
#   The contents of this file are subject to the terms and conditions defined in the License.
#   You may not use, modify, or distribute this file except in compliance with the License.
#
#   SPDX-License-Identifier: Apache-2.0
#
from collections.abc import Callable, Iterable
from functools import wraps
from itertools import repeat
from typing import Any

import numpy

__all__ = ["Translation", "Rotation", "Scaling", "Noise", "bin2float"]


class Translation:
    """Decorator that translates an individual before evaluation.

    The decorated function receives a plain list of translated values.
    After decoration, ``func.translate`` updates the translation vector.

    Args:
        vector: Translation values. Must have the same length as the
            individual.
    """

    vector: list[float]

    def __init__(self, vector: list[float]) -> None:
        """See the class docstring."""
        self.translate(vector)

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap an evaluation function with the translation.

        Args:
            func: Evaluation function to decorate.

        Returns:
            A callable that translates the individual, then calls
            ``func``. The wrapper has a ``translate`` method.
        """

        @wraps(func)
        def wrapper(individual: Any, *args: Any, **kwargs: Any) -> Any:
            translated = [v - t for v, t in zip(individual, self.vector, strict=False)]
            return func(translated, *args, **kwargs)

        decorated: Any = wrapper
        decorated.translate = self.translate
        return decorated

    def translate(self, vector: list[float]) -> None:
        """Update the translation vector.

        After decorating the evaluation function, this method is
        available on the function object.

        Args:
            vector: The translation vector.
        """
        self.vector = vector


class Rotation:
    """Decorator that rotates an individual before evaluation.

    The decorated function receives a plain ndarray of rotated values.
    After decoration, ``func.rotate`` updates the rotation matrix.

    Args:
        matrix: Orthogonal N-by-N rotation matrix, where N is the
            length of the individual.
    """

    matrix: numpy.ndarray

    def __init__(self, matrix: numpy.ndarray) -> None:
        """See the class docstring."""
        self.rotate(matrix)

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap an evaluation function with the rotation.

        Args:
            func: Evaluation function to decorate.

        Returns:
            A callable that rotates the individual, then calls
            ``func``. The wrapper has a ``rotate`` method.
        """

        @wraps(func)
        def wrapper(individual: Any, *args: Any, **kwargs: Any) -> Any:
            rotated = numpy.dot(self.matrix, individual)
            return func(rotated, *args, **kwargs)

        decorated: Any = wrapper
        decorated.rotate = self.rotate
        return decorated

    def rotate(self, matrix: numpy.ndarray) -> None:
        """Update the rotation matrix.

        After decorating the evaluation function, this method is
        available on the function object.

        Args:
            matrix: The rotation matrix.
        """
        self.matrix = numpy.linalg.inv(matrix)


class Scaling:
    """Decorator that scales an individual before evaluation.

    The decorated function receives a plain list of scaled values.
    After decoration, ``func.scale`` updates the scale factors.

    Args:
        factor: Scale factors. Must have the same length as the
            individual.
    """

    factor: tuple[float, ...]

    def __init__(self, factor: list[float]) -> None:
        """See the class docstring."""
        self.scale(factor)

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap an evaluation function with the scaling.

        Args:
            func: Evaluation function to decorate.

        Returns:
            A callable that scales the individual, then calls
            ``func``. The wrapper has a ``scale`` method.
        """

        @wraps(func)
        def wrapper(individual: Any, *args: Any, **kwargs: Any) -> Any:
            scaled = [v * f for v, f in zip(individual, self.factor, strict=False)]
            return func(scaled, *args, **kwargs)

        decorated: Any = wrapper
        decorated.scale = self.scale
        return decorated

    def scale(self, factor: list[float]) -> None:
        """Update the scale factors.

        After decorating the evaluation function, this method is
        available on the function object.

        Args:
            factor: The scale factor.
        """
        self.factor = tuple(1.0 / f for f in factor)


class Noise:
    """Decorator that adds noise to an evaluation result.

    Each noise generator is called without arguments. After
    decoration, ``func.add_noise`` updates the generators.

    Args:
        funcs: Noise generator callables. A single callable is
            applied to every result value. A list must match the
            result length. A ``None`` entry leaves that value
            unchanged.
    """

    rand_funcs: Iterable[Callable[..., Any] | None]

    def __init__(self, funcs: Callable[..., Any] | list[Callable[..., Any] | None]) -> None:
        """See the class docstring."""
        self.add_noise(funcs)

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap an evaluation function with noise on its result.

        Args:
            func: Evaluation function to decorate.

        Returns:
            A callable that calls ``func`` and adds noise to the
            result. The wrapper has an ``add_noise`` method.
        """

        @wraps(func)
        def wrapper(individual: Any, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
            result = func(individual, *args, **kwargs)
            if not isinstance(result, Iterable):
                result = (result,)
            noisy = []
            for r, f in zip(result, self.rand_funcs, strict=False):
                if f is None:
                    noisy.append(r)
                else:
                    noisy.append(r + f())
            return tuple(noisy)

        decorated: Any = wrapper
        decorated.add_noise = self.add_noise
        return decorated

    def add_noise(self, funcs: Callable[..., Any] | list[Callable[..., Any] | None]) -> None:
        """Update the noise generators.

        After decorating the evaluation function, this method is
        available on the function object.

        Args:
            funcs: The noise function or functions.
        """
        if callable(funcs) and not isinstance(funcs, list):
            self.rand_funcs = repeat(funcs)
        else:
            self.rand_funcs = funcs


def bin2float(min_: float, max_: float, n_bits: int) -> Callable[..., Any]:
    """Return a decorator that decodes a binary individual to floats.

    Each float uses ``n_bits`` bits and is mapped into
    ``[min_, max_]``. The decorated function receives the decoded
    float array.

    Args:
        min_: Lower bound of each decoded value.
        max_: Upper bound of each decoded value.
        n_bits: Bits used to encode each float.

    Returns:
        A decorator for an evaluation function.
    """

    def wrapper(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def wrapped(individual: Any, *args: Any, **kwargs: Any) -> Any:
            nelem = len(individual) // n_bits
            decoded = [0] * nelem
            for i in range(nelem):
                start = i * n_bits
                stop = i * n_bits + n_bits
                values = individual[start:stop]
                gene = int("".join(str(int(bool(bit))) for bit in values), 2)
                div = 2**n_bits - 1
                decoded[i] = min_ + ((gene / div) * (max_ - min_))
            return function(decoded, *args, **kwargs)

        return wrapped

    return wrapper

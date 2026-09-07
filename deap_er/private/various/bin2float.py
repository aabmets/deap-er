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
from collections.abc import Callable
from functools import wraps
from typing import Any

__all__: list[str] = ["bin2float"]


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
            div = 2**n_bits - 1
            span = max_ - min_
            for i in range(nelem):
                gene = 0
                start = i * n_bits
                for bit in individual[start : start + n_bits]:
                    gene = (gene << 1) | (1 if bit else 0)
                decoded[i] = min_ + ((gene / div) * span)
            return function(decoded, *args, **kwargs)

        return wrapped

    return wrapper

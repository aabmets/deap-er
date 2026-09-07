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
from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deap_er.private.typedefs import NumOrSeq

__all__: list[str] = ["broadcast_param", "require_positive_eta"]


def require_positive_eta(eta: float) -> None:
    """Require a strictly positive crowding degree.

    Args:
        eta: Crowding degree of a bounded SBX or polynomial operator.

    Raises:
        ValueError: If ``eta`` is not greater than 0.
    """
    if eta <= 0:
        raise ValueError("Argument 'eta' must be greater than 0.")


def broadcast_param(
    name: str, var: NumOrSeq, size: int, subject: str = "the individual"
) -> Sequence[int] | Sequence[float]:
    """Broadcast a scalar parameter or validate a per-gene sequence.

    Args:
        name: Argument name used in the error message.
        var: A single value or a sequence of per-gene values.
            Python numbers, ``numbers.Integral``, and ``numbers.Real``
            (including NumPy scalars) broadcast.
        size: Required number of values.
        subject: Noun phrase naming what ``size`` was measured from,
            used in the error message.

    Returns:
        A sequence of at least ``size`` values.

    Raises:
        ValueError: If ``var`` is a sequence shorter than ``size``.
    """
    if isinstance(var, int | float):
        return [var] * size
    # NumPy integers are Integral but not Python int.
    # NumPy float32 / float16 are Real but not Python float.
    if isinstance(var, Integral):
        return [int(var)] * size
    if isinstance(var, Real):
        return [float(var)] * size
    if len(var) < size:
        raise ValueError(
            f"Argument '{name}' must be at least the size of {subject}: {len(var)} < {size}"
        )
    return var

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
from collections.abc import Sequence

from deap_er.base.dtypes import NumOrSeq

__all__: list[str] = []


def _broadcast_param(
    name: str, var: NumOrSeq, size: int, subject: str = "the individual"
) -> Sequence[int] | Sequence[float]:
    """Broadcast a scalar parameter or validate a per-gene sequence.

    Args:
        name: Argument name used in the error message.
        var: A single value or a sequence of per-gene values.
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
    if len(var) < size:
        raise ValueError(
            f"Argument '{name}' must be at least the size of {subject}: {len(var)} < {size}"
        )
    return var

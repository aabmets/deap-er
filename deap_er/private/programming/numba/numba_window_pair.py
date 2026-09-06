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
from typing import Any

from . import numba_codes as codes
from .numba_window_pair_roll import roll_pair_stats

__all__: list[str] = ["apply_pair_window"]


def apply_pair_window(  # pragma: no cover
    op: int, rows: int, sp: int, stack: Any, scratch: Any, arg: int
) -> int:
    """Apply a two-input rolling covariance, correlation, or beta.

    Args:
        op: Opcode in the pair-window group.
        rows: Number of samples.
        sp: Current stack pointer.
        stack: Column-length workspace.
        scratch: Spare row of ``rows`` values.
        arg: Window length.

    Returns:
        The updated stack pointer.

    Raises:
        ValueError: If ``op`` is not a pair-window opcode.
    """
    if op not in (codes.ROLL_CORR, codes.ROLL_COV, codes.ROLL_BETA):
        raise ValueError(codes.UNKNOWN_OPCODE)
    roll_pair_stats(op, rows, sp, stack, scratch, arg)
    return sp - 1

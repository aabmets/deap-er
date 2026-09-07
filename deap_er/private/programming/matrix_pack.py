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
from typing import Any

import numpy

__all__: list[str] = ["as_matrix", "is_packed_matrix"]


def is_packed_matrix(columns: Any) -> bool:
    """Report whether ``columns`` is already a two-dimensional table.

    Args:
        columns: Column sequence or packed matrix passed to a tape runner.

    Returns:
        True when ``columns`` is a two-dimensional NumPy array.
    """
    return isinstance(columns, numpy.ndarray) and columns.ndim == 2


def as_matrix(columns: Sequence[Any], expected: int) -> numpy.ndarray:
    """Pack runner arguments into one contiguous column matrix.

    Args:
        columns: Either one two-dimensional matrix, or one array per
            column.
        expected: Number of columns the tape expects.

    Returns:
        A C-contiguous ``(n_rows, n_columns)`` ``float64`` matrix.

    Raises:
        ValueError: If the column count does not match the tape.
    """
    if len(columns) == 1 and is_packed_matrix(columns[0]):
        matrix = numpy.ascontiguousarray(columns[0], dtype=numpy.float64)
    else:
        if len(columns) != expected:
            raise ValueError(f"The tape expects {expected} columns, got {len(columns)}.")
        parts = [numpy.asarray(column, dtype=numpy.float64) for column in columns]
        matrix = numpy.ascontiguousarray(numpy.stack(parts, axis=1))
    if matrix.shape[1] != expected:
        raise ValueError(f"The tape expects {expected} columns, got {matrix.shape[1]}.")
    return matrix

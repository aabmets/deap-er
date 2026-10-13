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
from typing import Any, Literal

import numpy

__all__: list[str] = [
    "DISPATCH_FLOAT32",
    "MatrixPrecision",
    "as_matrix",
    "is_packed_matrix",
    "matrix_dtype",
]

type MatrixPrecision = Literal["float64", "float32"]

DISPATCH_FLOAT32 = (
    "precision='float32' cannot be combined with a dispatch kernel: "
    "USER_DISPATCH_SIGNATURE types the columns as float64. Use precision='float64'."
)

_DTYPES = {"float64": numpy.float64, "float32": numpy.float32}


def matrix_dtype(precision: MatrixPrecision) -> type[numpy.floating]:
    """Return the storage dtype of a packed column matrix.

    Args:
        precision: Either ``'float64'`` or ``'float32'``.

    Returns:
        ``numpy.float64`` or ``numpy.float32``.

    Raises:
        ValueError: If ``precision`` is not one of the two names.
    """
    if not isinstance(precision, str) or precision not in _DTYPES:
        raise ValueError(f"precision must be 'float64' or 'float32', got {precision!r}.")
    return _DTYPES[precision]


def is_packed_matrix(columns: Any) -> bool:
    """Report whether ``columns`` is already a two-dimensional table.

    Args:
        columns: Column sequence or packed matrix passed to a tape runner.

    Returns:
        True when ``columns`` is a two-dimensional NumPy array.
    """
    return isinstance(columns, numpy.ndarray) and columns.ndim == 2


def as_matrix(
    columns: Sequence[Any], expected: int, precision: MatrixPrecision = "float64"
) -> numpy.ndarray:
    """Pack runner arguments into one contiguous column matrix.

    Args:
        columns: Either one two-dimensional matrix, or one array per
            column.
        expected: Number of columns the tape expects.
        precision: Storage dtype of the matrix, ``'float64'`` or
            ``'float32'``. A matrix that already has it and is
            C-contiguous is used without a copy.

    Returns:
        A C-contiguous ``(n_rows, n_columns)`` matrix of ``precision``.

    Raises:
        ValueError: If the column count does not match the tape, or if
            ``precision`` is unknown.
    """
    dtype = matrix_dtype(precision)
    if len(columns) == 1 and is_packed_matrix(columns[0]):
        matrix = numpy.ascontiguousarray(columns[0], dtype=dtype)
    else:
        if len(columns) != expected:
            raise ValueError(f"The tape expects {expected} columns, got {len(columns)}.")
        parts = [numpy.asarray(column, dtype=dtype) for column in columns]
        matrix = numpy.ascontiguousarray(numpy.stack(parts, axis=1))
    if matrix.shape[1] != expected:
        raise ValueError(f"The tape expects {expected} columns, got {matrix.shape[1]}.")
    return matrix

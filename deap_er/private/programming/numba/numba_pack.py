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

import numpy

from ..tape import Tape

__all__: list[str] = ["pack_tapes"]


def pack_tapes(tapes: Sequence[Tape]) -> dict[str, numpy.ndarray | int]:
    """Concatenate tapes into the jagged streams the batch kernels read.

    Tape ``i`` owns ``opcodes[op_starts[i]:op_starts[i] + op_lens[i]]``
    (and the same slice of ``operands``) and
    ``constants[c_starts[i]:c_starts[i] + c_lens[i]]``, so each tape
    keeps indexing its own constant pool.

    Args:
        tapes: Tapes to pack. Must not be empty.

    Returns:
        The ``opcodes``, ``operands``, and ``constants`` streams, the
        ``op_starts``, ``op_lens``, ``c_starts``, and ``c_lens``
        offsets, the per-tape ``fills``, and the batch ``max_depth``.
    """
    op_lens = numpy.array([tape.opcodes.size for tape in tapes], dtype=numpy.int64)
    c_lens = numpy.array([tape.constants.size for tape in tapes], dtype=numpy.int64)
    op_starts = numpy.zeros(len(tapes), dtype=numpy.int64)
    c_starts = numpy.zeros(len(tapes), dtype=numpy.int64)
    op_starts[1:] = numpy.cumsum(op_lens[:-1])
    c_starts[1:] = numpy.cumsum(c_lens[:-1])
    return {
        "opcodes": numpy.concatenate([tape.opcodes for tape in tapes]).astype(
            numpy.int32, copy=False
        ),
        "operands": numpy.concatenate([tape.operands for tape in tapes]).astype(
            numpy.int32, copy=False
        ),
        "constants": numpy.concatenate([tape.constants for tape in tapes]).astype(
            numpy.float64, copy=False
        ),
        "op_starts": op_starts,
        "op_lens": op_lens,
        "c_starts": c_starts,
        "c_lens": c_lens,
        "fills": numpy.array([tape.fill for tape in tapes], dtype=numpy.float64),
        "max_depth": max(tape.depth for tape in tapes),
    }

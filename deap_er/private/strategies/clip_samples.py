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

from contextlib import suppress
from typing import Any

import numpy

__all__ = ["RAW_SAMPLE_ATTR", "raw_sample", "tag_raw_sample"]

RAW_SAMPLE_ATTR = "cma_raw_sample"


def tag_raw_sample(individual: Any, raw: numpy.ndarray, clipped: numpy.ndarray) -> Any:
    """Store the unclipped draw on ``individual`` when clipping moved it.

    ``bound_mode="clip"`` evaluates the clipped point, but the CMA
    update learns from the original draw. Updating from the clipped
    point would zero the variance across a box face.

    Args:
        individual: Individual built from ``clipped``.
        raw: The draw before clipping.
        clipped: ``raw`` clipped into the box.

    Returns:
        ``individual``, tagged in place when ``clipped`` differs from
        ``raw``. Types without instance attributes are left untagged.
    """
    if not numpy.array_equal(raw, clipped):
        with suppress(AttributeError):
            setattr(individual, RAW_SAMPLE_ATTR, raw)
    return individual


def raw_sample(individual: Any) -> numpy.ndarray:
    """Return the draw the CMA update should learn from.

    Args:
        individual: Individual made by a strategy's ``generate``.

    Returns:
        The stored unclipped draw, or the genes when the individual
        was not clipped.
    """
    return numpy.asarray(getattr(individual, RAW_SAMPLE_ATTR, individual), dtype=float)

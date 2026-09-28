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

from deap_er.private.operators.bounds import broadcast_param
from deap_er.private.strategies.common import update_bound_attrs

from .ephemeral_leaves import leaf_range

__all__: list[str] = ["BOUND_ATTRS", "box_strategy", "restore_bounds", "clipped_centroid"]

BOUND_ATTRS = ("low", "up", "bound_mode", "resample_limit")


def restore_bounds(strategy: Any, saved: dict[str, Any]) -> None:
    """Put the caller's bound attributes back after a boxed tune.

    Args:
        strategy: Strategy whose bound attributes were boxed.
        saved: Bound attributes the strategy had before boxing.
            Attributes missing here are removed again.
    """
    for name in BOUND_ATTRS:
        if name in saved:
            setattr(strategy, name, saved[name])
        elif hasattr(strategy, name):
            delattr(strategy, name)


def box_strategy(strategy: Any, nodes: Sequence[Any]) -> None:
    """Apply clip bounds from leaf ranges, tightening any caller box.

    Args:
        strategy: Strategy whose bound attributes are set in place.
        nodes: Numeric leaves aligned with the strategy coordinates.
    """
    box = _merged_box(strategy, nodes)
    if box is None:
        return
    lows, highs = box
    update_bound_attrs(strategy, {"low": lows, "up": highs, "bound_mode": "clip"})


def _merged_box(strategy: Any, nodes: Sequence[Any]) -> tuple[list[float], list[float]] | None:
    """Intersect caller ``low`` / ``up`` with per-leaf legal ranges."""
    leaf_lo, leaf_hi, has_leaf = _collect_leaf_ranges(nodes)
    caller_lo = getattr(strategy, "low", None)
    caller_hi = getattr(strategy, "up", None)
    if not has_leaf and caller_lo is None and caller_hi is None:
        return None
    if caller_lo is None and caller_hi is None:
        return _open_leaf_box(leaf_lo, leaf_hi)
    lows, highs = _broadcast_caller_box(caller_lo, caller_hi, len(nodes))
    return _intersect_boxes(lows, highs, leaf_lo, leaf_hi)


def _collect_leaf_ranges(
    nodes: Sequence[Any],
) -> tuple[list[float | None], list[float | None], bool]:
    """Return per-leaf ranges and whether any leaf is boxed."""
    leaf_lo = []
    leaf_hi = []
    has_leaf = False
    for node in nodes:
        low, high = leaf_range(node)
        leaf_lo.append(low)
        leaf_hi.append(high)
        if low is not None or high is not None:
            has_leaf = True
    return leaf_lo, leaf_hi, has_leaf


def _open_leaf_box(
    leaf_lo: Sequence[float | None],
    leaf_hi: Sequence[float | None],
) -> tuple[list[float], list[float]]:
    """Box from leaf ranges only; missing sides stay infinite."""
    lows = [-numpy.inf if value is None else float(value) for value in leaf_lo]
    highs = [numpy.inf if value is None else float(value) for value in leaf_hi]
    return lows, highs


def _broadcast_caller_box(
    caller_lo: Any,
    caller_hi: Any,
    dim: int,
) -> tuple[list[float], list[float]]:
    """Broadcast caller ``low`` / ``up`` to ``dim`` coordinates."""
    lows = [
        float(value)
        for value in broadcast_param("low", -numpy.inf if caller_lo is None else caller_lo, dim)
    ]
    highs = [
        float(value)
        for value in broadcast_param("up", numpy.inf if caller_hi is None else caller_hi, dim)
    ]
    return lows, highs


def _intersect_boxes(
    lows: list[float],
    highs: list[float],
    leaf_lo: Sequence[float | None],
    leaf_hi: Sequence[float | None],
) -> tuple[list[float], list[float]]:
    """Tighten each coordinate with the leaf range when both sides exist."""
    for index, low in enumerate(leaf_lo):
        if low is not None:
            lows[index] = _tighten_low(lows[index], low)
        high = leaf_hi[index]
        if high is not None:
            highs[index] = _tighten_high(highs[index], high)
    return lows, highs


def _tighten_low(current: float, leaf: float) -> float:
    """Raise a finite lower bound; replace an open side with the leaf."""
    if numpy.isfinite(current):
        return max(current, leaf)
    return float(leaf)


def _tighten_high(current: float, leaf: float) -> float:
    """Lower a finite upper bound; replace an open side with the leaf."""
    if numpy.isfinite(current):
        return min(current, leaf)
    return float(leaf)


def clipped_centroid(strategy: Any) -> numpy.ndarray:
    """Return ``strategy.centroid`` clipped to its current box.

    Args:
        strategy: Strategy that holds ``centroid`` and optional bounds.

    Returns:
        The clipped centroid as a float array.
    """
    vector = numpy.asarray(strategy.centroid, dtype=float)
    low = getattr(strategy, "low", None)
    up = getattr(strategy, "up", None)
    if low is None and up is None:
        return vector
    dim = len(vector)
    lo = numpy.asarray(broadcast_param("low", -numpy.inf if low is None else low, dim), dtype=float)
    hi = numpy.asarray(broadcast_param("up", numpy.inf if up is None else up, dim), dtype=float)
    return numpy.clip(vector, lo, hi)

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

import copy
from collections.abc import Callable, Sequence
from typing import Any

import numpy

from deap_er.private.operators.bounds import broadcast_param
from deap_er.private.strategies.common import update_bound_attrs
from deap_er.private.various.clone import clone_individual

from .compile_cache import expression_key
from .compilers import invalidate_compiled
from .ephemeral_leaves import assign_ephemerals, extract_ephemerals, leaf_range, numeric_leaves

__all__: list[str] = [
    "numeric_leaves",
    "extract_ephemerals",
    "assign_ephemerals",
    "tune_ephemerals",
]


def tune_ephemerals(
    individual: Any,
    strategy: Any,
    evaluate: Callable[[Any], Any] | None = None,
    n_gen: int = 5,
    *,
    evaluate_batch: Callable[[list[Any]], Any] | None = None,
    clone: Callable[[Any], Any] | None = None,
) -> Any:
    """Polish numeric leaves with a short boxed CMA run.

    Extracts ephemeral floats and ``Window`` ints, runs ``n_gen``
    ``generate`` / ``update`` steps on ``strategy``, writes the
    repaired centroid back, then invalidates fitness and the compile
    cache for the previous expression. Evaluation is the caller's
    ``evaluate`` on clones, or ``evaluate_batch`` on a pack of clones.
    Provide one of those callables; when both are set, the batch path
    is used.

    Args:
        individual: ``PrimitiveTree`` or ``SlimTree`` to tune in place.
        strategy: ``Strategy`` or ``StrategySeparable`` whose
            ``dim`` matches the leaf count.
        evaluate: ``callable(ind) ->`` fitness tuple. Optional when
            ``evaluate_batch`` is given.
        n_gen: Inner CMA generations. Default ``5``.
        evaluate_batch: Optional ``callable(inds) ->`` fitness tuples.
        clone: Individual copier. Defaults to ``clone_individual``.

    Returns:
        The same ``individual`` after write-back.

    Raises:
        ValueError: If ``n_gen < 1``, ``strategy.dim`` does not match
            the number of numeric leaves, or neither ``evaluate`` nor
            ``evaluate_batch`` is given.
    """
    if n_gen < 1:
        raise ValueError(f"n_gen must be at least 1, got {n_gen}.")
    leaves = numeric_leaves(individual)
    if not leaves:
        return individual
    if evaluate is None and evaluate_batch is None:
        raise ValueError("Provide evaluate or evaluate_batch.")
    if getattr(strategy, "dim", None) != len(leaves):
        raise ValueError(
            f"strategy.dim is {getattr(strategy, 'dim', None)}, "
            f"but the individual has {len(leaves)} numeric leaves."
        )
    strategy.centroid = numpy.asarray(extract_ephemerals(individual), dtype=float)
    _box_strategy(strategy, [tree[index] for tree, index in leaves])
    copier = clone_individual if clone is None else clone
    for _ in range(n_gen):
        trials = strategy.generate(_trial_init(individual))
        _score_trials(individual, trials, evaluate, evaluate_batch, copier)
        strategy.update(trials)
    old_keys = [expression_key(individual)]
    old_keys.extend(expression_key(tree) for tree, _ in leaves)
    assign_ephemerals(individual, _clipped_centroid(strategy).tolist())
    for key in dict.fromkeys(old_keys):
        invalidate_compiled(key)
    fitness = getattr(individual, "fitness", None)
    if fitness is not None and fitness.is_valid():
        del individual.fitness.values
    return individual


def _box_strategy(strategy: Any, nodes: Sequence[Any]) -> None:
    """Apply clip bounds from leaf ranges, tightening any caller box."""
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


def _clipped_centroid(strategy: Any) -> numpy.ndarray:
    """Return ``strategy.centroid`` clipped to its current box."""
    vector = numpy.asarray(strategy.centroid, dtype=float)
    low = getattr(strategy, "low", None)
    up = getattr(strategy, "up", None)
    if low is None and up is None:
        return vector
    dim = len(vector)
    lo = numpy.asarray(broadcast_param("low", -numpy.inf if low is None else low, dim), dtype=float)
    hi = numpy.asarray(broadcast_param("up", numpy.inf if up is None else up, dim), dtype=float)
    return numpy.clip(vector, lo, hi)


def _trial_init(individual: Any) -> Callable[[Any], list[Any]]:
    """Return an ``ind_init`` that copies ``individual.fitness``."""

    def factory(values: Any) -> _Trial:
        trial = _Trial(numpy.asarray(values, dtype=float).tolist())
        trial.fitness = copy.deepcopy(individual.fitness)
        if trial.fitness.is_valid():
            del trial.fitness.values
        return trial

    return factory


def _score_trials(
    individual: Any,
    trials: Sequence[Any],
    evaluate: Callable[[Any], Any] | None,
    evaluate_batch: Callable[[list[Any]], Any] | None,
    clone: Callable[[Any], Any],
) -> None:
    """Evaluate tree clones and copy fitness onto CMA trial vectors."""
    clones = []
    for trial in trials:
        replica = clone(individual)
        assign_ephemerals(replica, trial)
        clones.append(replica)
    if evaluate_batch is not None:
        fitnesses = evaluate_batch(clones)
    elif evaluate is not None:
        fitnesses = map(evaluate, clones)
    else:
        raise ValueError("Provide evaluate or evaluate_batch.")
    for trial, fit in zip(trials, fitnesses, strict=True):
        trial.fitness.values = fit


class _Trial(list[Any]):
    """CMA trial vector that carries a fitness slot."""

    fitness: Any

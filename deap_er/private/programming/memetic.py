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

from deap_er.private.various.clone import clone_individual

from .compile_cache import expression_key
from .compilers import invalidate_compiled
from .ephemeral_leaves import assign_ephemerals, extract_ephemerals, numeric_leaves
from .memetic_box import BOUND_ATTRS, box_strategy, clipped_centroid, restore_bounds

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
    repaired centroid back, then invalidates fitness, the compile
    cache, and matching ``EvalCache`` keys for the previous
    expression. Leaf ranges box the run only; the bound attributes of
    ``strategy`` are restored afterwards. Evaluation is the caller's
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
    saved = {name: getattr(strategy, name) for name in BOUND_ATTRS if hasattr(strategy, name)}
    try:
        box_strategy(strategy, [tree[index] for tree, index in leaves])
        copier = clone_individual if clone is None else clone
        for _ in range(n_gen):
            trials = strategy.generate(_trial_init(individual))
            _score_trials(individual, trials, evaluate, evaluate_batch, copier)
            strategy.update(trials)
        repaired = clipped_centroid(strategy).tolist()
    finally:
        restore_bounds(strategy, saved)
    old_keys = [expression_key(individual)]
    old_keys.extend(expression_key(tree) for tree, _ in leaves)
    assign_ephemerals(individual, repaired)
    for key in dict.fromkeys(old_keys):
        invalidate_compiled(key)
    fitness = getattr(individual, "fitness", None)
    if fitness is not None and fitness.is_valid():
        del individual.fitness.values
    return individual


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

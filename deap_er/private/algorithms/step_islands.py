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
from collections.abc import Callable, Hashable, Sequence

from deap_er.private.toolbox import Toolbox
from deap_er.private.typedefs import Individual

from .loop import evaluate_invalid

__all__: list[str] = ["step_islands"]


def _require_operator(toolbox: Toolbox, name: str) -> None:
    """Reject a deme toolbox that is missing a required alias.

    Args:
        toolbox: Deme toolbox about to step.
        name: Operator alias that must be registered.

    Raises:
        ValueError: If ``name`` is not registered on ``toolbox``.
    """
    if not hasattr(toolbox, name):
        raise ValueError(f"step_islands requires a '{name}' operator on each deme toolbox.")


def step_islands(
    demes: Sequence[tuple[Toolbox, list[Individual]]],
    migrate: Callable[[list[list[Individual]]], None] | None = None,
    *,
    eval_keys: Sequence[Hashable] | None = None,
) -> None:
    """Run one generation on unlike demes, then optionally migrate.

    Each deme is a ``(toolbox, population)`` pair. The toolbox must
    provide ``vary``, ``select``, and ``evaluate`` (or
    ``evaluate_batch``). The step is evaluate invalids, vary,
    evaluate the offspring, then replace the population with
    ``select(offspring, len(population))``. Populations are modified
    in place. A MAP-Elites island is a custom ``vary`` / ``select``
    pair that closes over an archive; this function does not call
    ``ea_map_elites`` and does not merge archives.

    ``migrate``, if given, receives the list of populations after
    every deme has stepped. Use ``mig_ring`` for a ring; this function
    does not pick a topology.

    Migrants keep their fitness when ``eval_keys`` is omitted or every
    key is equal. Distinct keys mean the destination's cases or matrix
    differ: immigrant fitness is cleared, including clones created by
    a replacement migration.

    Args:
        demes: ``(toolbox, population)`` pairs to step.
        migrate: Optional callable ``migrate(populations)``.
        eval_keys: Per-deme identity of the evaluation data. Length
            must match ``demes`` when given.

    Raises:
        ValueError: If a toolbox is missing ``vary`` or ``select``,
            or if ``eval_keys`` does not match the number of demes.
    """
    if eval_keys is not None and len(eval_keys) != len(demes):
        raise ValueError("eval_keys must have one entry per deme.")

    populations: list[list[Individual]] = []
    for toolbox, population in demes:
        _require_operator(toolbox, "vary")
        _require_operator(toolbox, "select")
        evaluate_invalid(toolbox, population)
        offspring = toolbox.vary(population)
        evaluate_invalid(toolbox, offspring)
        population[:] = toolbox.select(offspring, len(population))
        populations.append(population)

    if migrate is None:
        return

    if eval_keys is None or len(set(eval_keys)) <= 1:
        migrate(populations)
        return

    owner = {id(ind): key for key, pop in zip(eval_keys, populations, strict=True) for ind in pop}
    migrate(populations)
    for key, pop in zip(eval_keys, populations, strict=True):
        for ind in pop:
            if owner.get(id(ind)) != key:
                del ind.fitness.values

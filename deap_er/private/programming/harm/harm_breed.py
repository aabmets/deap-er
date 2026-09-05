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

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from deap_er.private.toolbox import Toolbox

if TYPE_CHECKING:
    from deap_er.private.typedefs import GPIndividual
from deap_er.private.various.rng import rng

__all__: list[str] = ["append_if_accepted", "mate_pair", "mutate_or_clone", "breed_into", "produce"]


def _accept_all(_size: int) -> bool:
    """Accept an individual of any size.

    Args:
        _size: Size of the candidate individual, ignored.

    Returns:
        Always True.
    """
    return True


def append_if_accepted(
    produced_pop: list[Any],
    produced_pop_sizes: list[int],
    aspirant: GPIndividual,
    accept_func: Callable[[int], bool],
) -> None:
    """Append ``aspirant`` when it passes ``accept_func``.

    Args:
        produced_pop: Accumulator for accepted individuals.
        produced_pop_sizes: Accumulator for their sizes.
        aspirant: Candidate individual.
        accept_func: Predicate on candidate size.
    """
    if accept_func(len(aspirant)):
        produced_pop.append(aspirant)
        produced_pop_sizes.append(len(aspirant))


def mate_pair(toolbox: Toolbox, population: list[GPIndividual]) -> tuple[Any, Any]:
    """Crossover two cloned parents from ``population``.

    Args:
        toolbox: Toolbox with ``mate``, ``clone``, and ``select``.
        population: Individuals to breed from.

    Returns:
        The two children, with fitness invalidated.
    """
    child1, child2 = toolbox.mate(*map(toolbox.clone, toolbox.select(population, 2)))
    del child1.fitness.values, child2.fitness.values
    return child1, child2


def mutate_or_clone(
    toolbox: Toolbox,
    population: list[GPIndividual],
    op_random: float,
    cx_prob: float,
    mut_prob: float,
) -> Any:
    """Reproduce one parent, optionally mutating it.

    Args:
        toolbox: Toolbox with ``clone``, ``select``, and ``mutate``.
        population: Individuals to breed from.
        op_random: Draw used to choose the operator.
        cx_prob: Crossover probability already rejected.
        mut_prob: Probability of mutating the clone.

    Returns:
        The child individual.
    """
    child = toolbox.clone(toolbox.select(population, 1)[0])
    if op_random - cx_prob < mut_prob:
        child = toolbox.mutate(child)[0]
        del child.fitness.values
    return child


def breed_into(
    toolbox: Toolbox,
    population: list[GPIndividual],
    produced_pop: list[Any],
    produced_pop_sizes: list[int],
    count: int,
    cx_prob: float,
    mut_prob: float,
    accept_func: Callable[[int], bool],
) -> None:
    """Breed one operator draw into ``produced_pop``.

    Args:
        toolbox: Toolbox with the variation operators.
        population: Individuals to breed from.
        produced_pop: Accumulator for accepted individuals.
        produced_pop_sizes: Accumulator for their sizes.
        count: Target number of accepted individuals.
        cx_prob: Probability of producing a child by crossover.
        mut_prob: Probability of producing a child by mutation.
        accept_func: Predicate on candidate size.
    """
    op_random = rng.random()
    if op_random < cx_prob:
        child1, child2 = mate_pair(toolbox, population)
        append_if_accepted(produced_pop, produced_pop_sizes, child1, accept_func)
        if len(produced_pop) < count:
            append_if_accepted(produced_pop, produced_pop_sizes, child2, accept_func)
        return
    child = mutate_or_clone(toolbox, population, op_random, cx_prob, mut_prob)
    append_if_accepted(produced_pop, produced_pop_sizes, child, accept_func)


def produce(
    toolbox: Toolbox,
    population: list[GPIndividual],
    count: int,
    cx_prob: float,
    mut_prob: float,
    pick_from: list[GPIndividual] | None = None,
    accept_func: Callable[[int], bool] = _accept_all,
) -> tuple[list[GPIndividual], list[int]]:
    """Produce individuals until ``count`` of them pass ``accept_func``.

    Candidates are taken from ``pick_from`` while it lasts, and are
    otherwise bred from ``population`` by crossover, mutation, or
    reproduction.

    Args:
        toolbox: Toolbox with the variation operators.
        population: Individuals to breed from.
        count: Number of individuals to produce.
        cx_prob: Probability of producing a child by crossover.
        mut_prob: Probability of producing a child by mutation.
        pick_from: Optional pool of ready-made candidates, consumed
            from the end.
        accept_func: Predicate on candidate size.

    Returns:
        The produced individuals and their sizes.
    """
    if pick_from is None:
        pick_from = []

    produced_pop: list[Any] = []
    produced_pop_sizes: list[int] = []

    while len(produced_pop) < count:
        if pick_from:
            append_if_accepted(produced_pop, produced_pop_sizes, pick_from.pop(), accept_func)
            continue
        breed_into(
            toolbox,
            population,
            produced_pop,
            produced_pop_sizes,
            count,
            cx_prob,
            mut_prob,
            accept_func,
        )

    return produced_pop, produced_pop_sizes

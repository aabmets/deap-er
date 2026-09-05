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
from collections.abc import Callable
from typing import Any

from deap_er.private.various.rng import rng

from .generators import gen_grow
from .primitives.primitive_nodes import Terminal
from .primitives.primitive_set_typed import PrimitiveSetTyped

__all__: list[str] = ["mut_semantic", "cx_semantic"]


def _check(p_set: PrimitiveSetTyped, op: str) -> None:
    """Require the semantic operators ``lf``, ``mul``, ``add``, and ``sub``.

    Args:
        p_set: Primitive set that must contain those names.
        op: Operation label used in the error message.

    Raises:
        TypeError: If any required name is missing from ``p_set.mapping``.
    """
    for func in ["lf", "mul", "add", "sub"]:
        if func not in p_set.mapping:
            raise TypeError(f"A '{func}' function is required to perform semantic '{op}'.")


def mut_semantic(
    individual: list[Any],
    prim_set: PrimitiveSetTyped,
    min_depth: int = 2,
    max_depth: int = 6,
    gen_func: Callable[..., Any] | None = None,
    mut_step: float | None = None,
) -> tuple[list[Any]]:
    """Mutate an individual by a semantic mutation.

    Args:
        individual: Individual to mutate.
        prim_set: Primitive set used to build the random trees.
        min_depth: Minimum depth of each random tree.
        max_depth: Maximum depth of each random tree.
        gen_func: Tree generator. Defaults to ``gen_grow``.
        mut_step: Mutation step. Drawn uniformly from ``[0, 2]``
            when omitted.

    Returns:
        A one-element tuple containing the mutated individual.
    """
    _check(prim_set, "mutation")

    if gen_func is None:
        gen_func = gen_grow

    if mut_step is None:
        mut_step = rng.uniform(0, 2)

    tr1 = gen_func(prim_set, min_depth, max_depth)
    tr2 = gen_func(prim_set, min_depth, max_depth)

    tr1.insert(0, prim_set.mapping["lf"])
    tr2.insert(0, prim_set.mapping["lf"])

    new_ind = individual
    new_ind.insert(0, prim_set.mapping["add"])
    new_ind.append(prim_set.mapping["mul"])

    mutation_step = Terminal(mut_step, False, object)
    new_ind.append(mutation_step)
    new_ind.append(prim_set.mapping["sub"])

    new_ind.extend(tr1)
    new_ind.extend(tr2)

    return (new_ind,)


def cx_semantic(
    ind1: list[Any],
    ind2: list[Any],
    prim_set: PrimitiveSetTyped,
    min_depth: int = 2,
    max_depth: int = 6,
    gen_func: Callable[..., Any] = gen_grow,
) -> tuple[list[Any], list[Any]]:
    """Mate two individuals by a semantic crossover.

    Args:
        ind1: First individual to mate.
        ind2: Second individual to mate.
        prim_set: Primitive set used to build the random tree.
        min_depth: Minimum depth of the random tree.
        max_depth: Maximum depth of the random tree.
        gen_func: Tree generator. Defaults to ``gen_grow``.

    Returns:
        The two individuals after crossover.
    """
    _check(prim_set, "crossover")

    tr = gen_func(prim_set, min_depth, max_depth)
    tr.insert(0, prim_set.mapping["lf"])

    def create_ind(ind: list[Any], ind_ext: list[Any]) -> list[Any]:
        new_ind = ind
        new_ind.insert(0, prim_set.mapping["mul"])
        new_ind.insert(0, prim_set.mapping["add"])
        new_ind.extend(tr)
        new_ind.append(prim_set.mapping["mul"])
        new_ind.append(prim_set.mapping["sub"])
        new_ind.append(Terminal(1.0, False, object))
        new_ind.extend(tr)
        new_ind.extend(ind_ext)
        return new_ind

    parent1 = list(ind1)
    parent2 = list(ind2)
    new_ind1 = create_ind(ind1, parent2)
    new_ind2 = create_ind(ind2, parent1)
    return new_ind1, new_ind2

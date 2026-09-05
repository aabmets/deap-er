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
from inspect import isclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deap_er.private.typedefs import GPIndividual, GPMutant
from deap_er.private.various.rng import rng

from .generators import choose_weighted
from .primitives.primitive_nodes import Ephemeral, Primitive
from .primitives.primitive_set_typed import PrimitiveSetTyped

__all__: list[str] = [
    "mut_uniform",
    "mut_node_replacement",
    "mut_ephemeral",
    "mut_insert",
    "mut_shrink",
]


def mut_uniform(
    individual: GPIndividual, expr: Callable[..., Any], prim_set: PrimitiveSetTyped
) -> GPMutant:
    """Replace a random subtree with an expression from ``expr``.

    Args:
        individual: GP tree to mutate.
        expr: Callable that returns a random subtree.
        prim_set: Primitive set passed to ``expr``.

    Returns:
        A one-element tuple containing the mutated individual.
    """
    index = rng.randrange(len(individual))
    i_slice = individual.search_subtree(index)
    ret_type = individual[index].ret
    individual[i_slice] = expr(prim_set=prim_set, ret_type=ret_type)
    return (individual,)


def mut_node_replacement(individual: GPIndividual, prim_set: PrimitiveSetTyped) -> GPMutant:
    """Replace a random node with a compatible node from ``prim_set``.

    Args:
        individual: GP tree to mutate.
        prim_set: Primitive set to sample the replacement from.

    Returns:
        A one-element tuple containing the mutated individual.
    """
    if len(individual) < 2:
        return (individual,)

    index = rng.randrange(1, len(individual))
    node = individual[index]

    if node.arity == 0:
        term = rng.choice(prim_set.terminals[node.ret])
        if isclass(term):
            term = term()
        individual[index] = term
    else:
        node_ret = prim_set.primitives[node.ret]
        prims = [p for p in node_ret if p.args == node.args]
        individual[index] = choose_weighted(prims)

    return (individual,)


def mut_ephemeral(individual: GPIndividual, mode: str = "all") -> GPMutant:
    """Resample one or all ephemeral constants in the tree.

    Args:
        individual: GP tree to mutate.
        mode: ``'one'`` to replace a single random ephemeral, or
            ``'all'`` to replace every ephemeral.

    Returns:
        A one-element tuple containing the mutated individual.

    Raises:
        ValueError: If ``mode`` is not ``'one'`` or ``'all'``.
    """
    if mode not in ["one", "all"]:
        raise ValueError("Mode must be one of 'one' or 'all'.")

    ephemera_idx = []
    for index, node in enumerate(individual):
        if isinstance(node, Ephemeral):
            ephemera_idx.append(index)

    if len(ephemera_idx) > 0:
        if mode == "one":
            ephemera_idx = (rng.choice(ephemera_idx),)

        for i in ephemera_idx:
            individual[i] = type(individual[i])()

    return (individual,)


def mut_insert(individual: GPIndividual, prim_set: PrimitiveSetTyped) -> GPMutant:
    """Insert a new primitive branch at a random position.

    Args:
        individual: GP tree to mutate.
        prim_set: Primitive set used to choose the inserted branch.

    Returns:
        A one-element tuple containing the mutated individual.
    """
    index = rng.randrange(len(individual))
    node = individual[index]
    slice_ = individual.search_subtree(index)

    primitives = []
    for p in prim_set.primitives[node.ret]:
        if node.ret in p.args:
            primitives.append(p)

    if len(primitives) == 0:
        return (individual,)

    new_node = choose_weighted(primitives)
    new_subtree = [None] * len(new_node.args)

    choices = []
    for i, a in enumerate(new_node.args):
        if a == node.ret:
            choices.append(i)
    position = rng.choice(choices)

    for i, arg_type in enumerate(new_node.args):
        if i != position:
            term = rng.choice(prim_set.terminals[arg_type])
            if isclass(term):
                term = term()
            new_subtree[i] = term

    new_subtree[position : position + 1] = individual[slice_]
    new_subtree.insert(0, new_node)
    individual[slice_] = new_subtree

    return (individual,)


def mut_shrink(individual: GPIndividual) -> GPMutant:
    """Replace a random branch with one of its arguments.

    Args:
        individual: GP tree to mutate.

    Returns:
        A one-element tuple containing the mutated individual.
    """
    if len(individual) < 3 or individual.height <= 1:
        return (individual,)

    i_prims = []
    for i, node in enumerate(individual[1:], 1):
        if isinstance(node, Primitive) and node.ret in node.args:
            i_prims.append((i, node))

    if len(i_prims) != 0:
        index, prim = rng.choice(i_prims)
        choices = []
        for i, type_ in enumerate(prim.args):
            if type_ == prim.ret:
                choices.append(i)
        arg_idx = rng.choice(choices)
        r_index = index + 1
        subtree = []
        for _ in range(arg_idx + 1):
            r_slice = individual.search_subtree(r_index)
            subtree = individual[r_slice]
            r_index += len(subtree)

        i_slice = individual.search_subtree(index)
        individual[i_slice] = subtree

    return (individual,)

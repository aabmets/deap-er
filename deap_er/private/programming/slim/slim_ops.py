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
from typing import Any

from deap_er.private.various.rng import rng

from ..generators import gen_grow
from ..primitives.primitive_set_typed import PrimitiveSetTyped
from ..semantic import _check, build_sig2_delta
from .slim_tree import SlimTree

__all__: list[str] = [
    "mut_slim_inflate",
    "mut_slim_deflate",
    "mut_slim",
    "cx_slim_donor",
]


def _as_slim(individual: SlimTree | list[Any]) -> SlimTree:
    """Return ``individual`` as a ``SlimTree``.

    Args:
        individual: SLIM genotype or standard tree to coerce.

    Returns:
        The same ``SlimTree`` or a wrapper around a standard tree.
    """
    return SlimTree.from_tree(individual)


def _pick_donor_index(slim1: SlimTree, slim2: SlimTree, best_donor: bool) -> int:
    """Choose which parent donates a delta block.

    Args:
        slim1: First parent.
        slim2: Second parent.
        best_donor: When both fitnesses are valid, pick the better parent.

    Returns:
        ``0`` when ``slim1`` is the donor, otherwise ``1``.
    """
    if (
        best_donor
        and hasattr(slim1, "fitness")
        and hasattr(slim2, "fitness")
        and slim1.fitness.is_valid()
        and slim2.fitness.is_valid()
    ):
        fitness1 = slim1.fitness
        fitness2 = slim2.fitness
        weight = fitness1.weights[0]
        value1 = fitness1.values[0]
        value2 = fitness2.values[0]
        if value1 != value2:
            better_first = value1 > value2 if weight > 0 else value1 < value2
            return 0 if better_first else 1
    return rng.randint(0, 1)


def mut_slim_inflate(
    individual: SlimTree,
    prim_set: PrimitiveSetTyped,
    min_depth: int = 2,
    max_depth: int = 6,
    gen_func: Callable[..., Any] | None = None,
    mut_step: float | None = None,
) -> tuple[SlimTree]:
    """Append a geometric semantic delta block (SLIM inflate mutation).

    Args:
        individual: SLIM genotype to mutate.
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
    slim = _as_slim(individual)
    if gen_func is None:
        gen_func = gen_grow
    if mut_step is None:
        mut_step = rng.uniform(0, 2)
    slim.deltas.append(build_sig2_delta(prim_set, min_depth, max_depth, gen_func, mut_step))
    return (slim,)


def mut_slim_deflate(
    individual: SlimTree,
    prim_set: PrimitiveSetTyped,
) -> tuple[SlimTree]:
    """Remove a random delta block (SLIM deflate mutation).

    The head tree is never removed. When there are no delta blocks,
    the individual is returned unchanged.

    Args:
        individual: SLIM genotype to mutate.
        prim_set: Primitive set checked for required semantic operators.

    Returns:
        A one-element tuple containing the mutated individual.
    """
    _check(prim_set, "mutation")
    slim = _as_slim(individual)
    if not slim.deltas:
        return (slim,)
    slim.deltas.pop(rng.randrange(len(slim.deltas)))
    return (slim,)


def mut_slim(
    individual: SlimTree,
    prim_set: PrimitiveSetTyped,
    *,
    inflate_prob: float = 0.3,
    min_depth: int = 2,
    max_depth: int = 6,
    gen_func: Callable[..., Any] | None = None,
    mut_step: float | None = None,
) -> tuple[SlimTree]:
    """Apply inflate or deflate mutation with fixed probability.

    Args:
        individual: SLIM genotype to mutate.
        prim_set: Primitive set used by the chosen mutation.
        inflate_prob: Probability of inflate mutation. Deflate is used
            otherwise.
        min_depth: Minimum depth of random trees for inflate.
        max_depth: Maximum depth of random trees for inflate.
        gen_func: Tree generator for inflate. Defaults to ``gen_grow``.
        mut_step: Mutation step for inflate. Drawn uniformly from
            ``[0, 2]`` when omitted.

    Returns:
        A one-element tuple containing the mutated individual.
    """
    if rng.random() < inflate_prob:
        return mut_slim_inflate(
            individual,
            prim_set,
            min_depth=min_depth,
            max_depth=max_depth,
            gen_func=gen_func,
            mut_step=mut_step,
        )
    return mut_slim_deflate(individual, prim_set)


def cx_slim_donor(
    ind1: SlimTree | list[Any],
    ind2: SlimTree | list[Any],
    prim_set: PrimitiveSetTyped,
    *,
    best_donor: bool = True,
) -> tuple[SlimTree, SlimTree]:
    """Mate two SLIM individuals by donor crossover (XODn / XOBDn).

    One delta block moves from the donor parent to the receiver.
    Offspring sizes stay close to the parents because one block is
    removed and one is added. When ``best_donor`` is True and both
    parents have valid fitness, the fitter parent donates (XOBDn);
    otherwise the donor is chosen uniformly (XODn). When the donor
    has no delta blocks, the parents are returned unchanged.

    Args:
        ind1: First parent.
        ind2: Second parent.
        prim_set: Primitive set checked for required semantic operators.
        best_donor: Prefer the fitter parent as donor when fitness is
            valid on both parents.

    Returns:
        The two parents after crossover.
    """
    _check(prim_set, "crossover")
    slim1 = _as_slim(ind1)
    slim2 = _as_slim(ind2)
    donor_idx = _pick_donor_index(slim1, slim2, best_donor)
    donor = slim1 if donor_idx == 0 else slim2
    receiver = slim2 if donor_idx == 0 else slim1
    if not donor.deltas:
        return slim1, slim2
    block = donor.deltas.pop(rng.randrange(len(donor.deltas)))
    receiver.deltas.append(block)
    return slim1, slim2

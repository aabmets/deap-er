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


def _require_slim(individual: Any, op: str) -> SlimTree:
    """Return ``individual`` when it is already a ``SlimTree``.

    Args:
        individual: Candidate SLIM genotype.
        op: Operation label used in the error message.

    Returns:
        The same ``SlimTree`` instance.

    Raises:
        TypeError: If ``individual`` is not a ``SlimTree``.
    """
    if not isinstance(individual, SlimTree):
        raise TypeError(
            f"SLIM {op} requires a SlimTree individual; got {type(individual).__name__}."
        )
    return individual


def _fitness_is_valid(fitness: Any) -> bool:
    """Return whether ``fitness`` has usable objective values.

    Args:
        fitness: Fitness object attached to a parent.

    Returns:
        True when comparison is safe for donor selection.
    """
    is_valid = getattr(fitness, "is_valid", None)
    if callable(is_valid):
        return bool(is_valid())
    if hasattr(fitness, "valid"):
        return bool(fitness.valid)
    values = getattr(fitness, "values", ())
    weights = getattr(fitness, "weights", ())
    return bool(values) and bool(weights) and len(values) == len(weights)


def _fitness_is_better(first: Any, second: Any) -> bool:
    """Return whether ``first`` is strictly better than ``second``.

    Args:
        first: Candidate fitter parent.
        second: Other parent.

    Returns:
        True when ``first`` should be preferred as donor.
    """
    greater = getattr(first, "__gt__", None)
    if callable(greater):
        result = greater(second)
        if result is not NotImplemented:
            return bool(result)
    first_wvalues = getattr(first, "wvalues", ())
    second_wvalues = getattr(second, "wvalues", ())
    if first_wvalues and second_wvalues:
        return first_wvalues > second_wvalues
    first_values = getattr(first, "values", ())
    second_values = getattr(second, "values", ())
    first_weights = getattr(first, "weights", ())
    if (
        first_values
        and second_values
        and first_weights
        and len(first_values) == len(second_values) == len(first_weights)
    ):
        first_score = tuple(
            value * weight for value, weight in zip(first_values, first_weights, strict=True)
        )
        second_score = tuple(
            value * weight for value, weight in zip(second_values, first_weights, strict=True)
        )
        return bool(first_score > second_score)
    return False


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
        and _fitness_is_valid(slim1.fitness)
        and _fitness_is_valid(slim2.fitness)
    ):
        if _fitness_is_better(slim1.fitness, slim2.fitness):
            return 0
        if _fitness_is_better(slim2.fitness, slim1.fitness):
            return 1
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
        individual: SLIM genotype to mutate in place.
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
    slim = _require_slim(individual, "inflate mutation")
    if gen_func is None:
        gen_func = gen_grow
    if mut_step is None:
        mut_step = rng.uniform(0, 2)
    slim.deltas.append(build_sig2_delta(prim_set, min_depth, max_depth, gen_func, mut_step))
    return (slim,)


def mut_slim_deflate(
    individual: SlimTree,
) -> tuple[SlimTree]:
    """Remove a random delta block (SLIM deflate mutation).

    The head tree is never removed. When there are no delta blocks,
    the individual is returned unchanged.

    Args:
        individual: SLIM genotype to mutate in place.

    Returns:
        A one-element tuple containing the mutated individual.
    """
    slim = _require_slim(individual, "deflate mutation")
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
        individual: SLIM genotype to mutate in place.
        prim_set: Primitive set used by inflate mutation.
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
    return mut_slim_deflate(individual)


def cx_slim_donor(
    ind1: SlimTree,
    ind2: SlimTree,
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
        ind1: First parent ``SlimTree``.
        ind2: Second parent ``SlimTree``.
        prim_set: Primitive set checked for required semantic operators.
        best_donor: Prefer the fitter parent as donor when fitness is
            valid on both parents.

    Returns:
        The two parents after crossover.
    """
    _check(prim_set, "crossover")
    slim1 = _require_slim(ind1, "donor crossover")
    slim2 = _require_slim(ind2, "donor crossover")
    donor_idx = _pick_donor_index(slim1, slim2, best_donor)
    donor = slim1 if donor_idx == 0 else slim2
    receiver = slim2 if donor_idx == 0 else slim1
    if donor.deltas:
        block = donor.deltas.pop(rng.randrange(len(donor.deltas)))
        receiver.deltas.append(block)
    return slim1, slim2

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
from collections import defaultdict
from collections.abc import Callable, Collection
from functools import partial
from operator import eq, lt

from deap_er.rng import rng

from .typedefs import GPIndividual, GPMates

__all__ = ["cx_one_point", "cx_one_point_leaf_biased"]


def _collect_indices(
    individual: GPIndividual, arity_op: Callable[[int], bool] | None = None
) -> defaultdict[type, list[int]]:
    """Group the node indices of an individual by return type.

    The root is never a candidate, so indexing starts at one.

    Args:
        individual: Individual to scan.
        arity_op: Optional predicate on node arity. When given, only
            nodes whose arity satisfies it are collected.

    Returns:
        A mapping of return type to the indices of matching nodes.
    """
    types: defaultdict[type, list[int]] = defaultdict(list)
    for idx, node in enumerate(individual[1:], 1):
        if arity_op is None or arity_op(node.arity):
            types[node.ret].append(idx)
    return types


def _swap_subtrees(
    ind1: GPIndividual,
    ind2: GPIndividual,
    types1: defaultdict[type, list[int]],
    types2: defaultdict[type, list[int]],
    common_types: Collection[type],
) -> None:
    """Swap one randomly chosen subtree of a shared return type.

    Both individuals are modified in place. Nothing happens when the
    two individuals share no return type.

    Args:
        ind1: First individual to mate.
        ind2: Second individual to mate.
        types1: Indices of ``ind1`` grouped by return type.
        types2: Indices of ``ind2`` grouped by return type.
        common_types: Return types present in both individuals.
    """
    if len(common_types) == 0:
        return

    type_ = rng.choice(list(common_types))

    index1 = rng.choice(types1[type_])
    index2 = rng.choice(types2[type_])
    slice1 = ind1.search_subtree(index1)
    slice2 = ind2.search_subtree(index2)
    ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]


def cx_one_point(ind1: GPIndividual, ind2: GPIndividual) -> GPMates:
    """Exchange a random subtree between two individuals.

    A crossover point is chosen in each tree and the subtrees rooted
    there are swapped. Individuals shorter than two nodes are returned
    unchanged.

    Args:
        ind1: First individual to mate.
        ind2: Second individual to mate.

    Returns:
        The two individuals after subtree exchange.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2

    types1 = _collect_indices(ind1)
    types2 = _collect_indices(ind2)
    common_types = set(types1.keys()).intersection(set(types2.keys()))

    _swap_subtrees(ind1, ind2, types1, types2, common_types)

    return ind1, ind2


def cx_one_point_leaf_biased(ind1: GPIndividual, ind2: GPIndividual, term_prob: float) -> GPMates:
    """Exchange a random subtree, biased toward terminals.

    Same as one-point crossover, except each parent independently
    selects a terminal as the crossover point with probability
    ``term_prob``.

    Args:
        ind1: First individual to mate.
        ind2: Second individual to mate.
        term_prob: Probability of choosing a terminal as the
            crossover point.

    Returns:
        The two individuals after subtree exchange.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2

    terminal_op = partial(eq, 0)
    primitive_op = partial(lt, 0)
    arity_op1 = terminal_op if rng.random() < term_prob else primitive_op
    arity_op2 = terminal_op if rng.random() < term_prob else primitive_op

    types1 = _collect_indices(ind1, arity_op1)
    types2 = _collect_indices(ind2, arity_op2)
    common_types = set(types1.keys()).intersection(set(types2.keys()))

    _swap_subtrees(ind1, ind2, types1, types2, common_types)

    return ind1, ind2

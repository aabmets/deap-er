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

from collections import defaultdict
from collections.abc import Callable, Collection, Sequence
from functools import partial
from operator import eq, lt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deap_er.private.typedefs import GPIndividual, GPMates
from deap_er.private.various.rng import rng

from .tape_lower import child_indices

__all__: list[str] = [
    "common_type_candidates",
    "path_from_root",
    "index_at_path",
    "swap_at",
    "cx_homologous",
    "cx_one_point",
    "cx_one_point_leaf_biased",
]


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


def common_type_candidates(
    ind1: GPIndividual, ind2: GPIndividual
) -> tuple[defaultdict[type, list[int]], defaultdict[type, list[int]], list[type]]:
    """Group crossover candidates in both trees and return shared types.

    Shared types keep their first-occurrence order in ``ind1``. A ``set``
    would order classes by ``id()``, so one seed would draw a different
    type in every process.

    Args:
        ind1: First individual.
        ind2: Second individual.

    Returns:
        Non-root indices of each tree grouped by return type, and the
        return types present in both.
    """
    types1 = _collect_indices(ind1)
    types2 = _collect_indices(ind2)
    common_types = [type_ for type_ in types1 if type_ in types2]
    return types1, types2, common_types


def path_from_root(children: Sequence[Sequence[int]], index: int) -> tuple[int, ...]:
    """Return left-to-right child ranks from the root to ``index``.

    Args:
        children: Child indices of every node, as ``child_indices`` returns.
        index: Node whose path is wanted.

    Returns:
        Child ranks from the root down to ``index``. Empty for the root.
    """
    parents = {
        kid: (node, rank) for node, kids in enumerate(children) for rank, kid in enumerate(kids)
    }
    path: list[int] = []
    while index != 0:
        index, rank = parents[index]
        path.append(rank)
    return tuple(reversed(path))


def index_at_path(children: Sequence[Sequence[int]], path: Sequence[int]) -> int | None:
    """Resolve the node index at ``path``.

    Args:
        children: Child indices of every node, as ``child_indices`` returns.
        path: Child ranks from the root.

    Returns:
        The node index, or ``None`` when the path does not exist.
    """
    index = 0
    for step in path:
        kids = children[index]
        if step >= len(kids):
            return None
        index = int(kids[step])
    return index


def swap_at(ind1: GPIndividual, ind2: GPIndividual, index1: int, index2: int) -> None:
    """Swap the subtrees rooted at ``index1`` and ``index2`` in place.

    Args:
        ind1: First individual.
        ind2: Second individual.
        index1: Subtree root in ``ind1``.
        index2: Subtree root in ``ind2``.
    """
    slice1 = ind1.search_subtree(index1)
    slice2 = ind2.search_subtree(index2)
    ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]


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
    swap_at(ind1, ind2, rng.choice(types1[type_]), rng.choice(types2[type_]))


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

    types1, types2, common_types = common_type_candidates(ind1, ind2)
    _swap_subtrees(ind1, ind2, types1, types2, common_types)
    return ind1, ind2


def cx_homologous(ind1: GPIndividual, ind2: GPIndividual) -> GPMates:
    """Exchange subtrees at the same root-to-node path when types match.

    A crossover point is chosen uniformly in ``ind1`` (never the root).
    The same child-index path is resolved in ``ind2``. When both nodes
    share a return type, their subtrees are swapped. Otherwise the
    operator falls back to random type-matched one-point crossover, the
    same contract as :func:`cx_one_point`.

    Args:
        ind1: First individual to mate. Its crossover point anchors the
            homologous path.
        ind2: Second individual to mate.

    Returns:
        The two individuals after subtree exchange.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2

    children1 = child_indices(list(ind1))
    index1 = int(rng.integers(1, len(ind1)))
    path = path_from_root(children1, index1)
    index2 = index_at_path(child_indices(list(ind2)), path)

    if index2 is not None and ind1[index1].ret == ind2[index2].ret:
        swap_at(ind1, ind2, index1, index2)
    else:
        types1, types2, common_types = common_type_candidates(ind1, ind2)
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
    common_types = [type_ for type_ in types1 if type_ in types2]
    _swap_subtrees(ind1, ind2, types1, types2, common_types)

    return ind1, ind2

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

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from deap_er.private.programming.primitives.primitive_set_typed import PrimitiveSetTyped
    from deap_er.private.typedefs import GPIndividual, GPMates
from deap_er.private.programming.crossover import (
    _common_type_candidates,
    _swap_at,
)
from deap_er.private.programming.primitives.primitive_tree import PrimitiveTree
from deap_er.private.programming.tape_batch import interpret_tapes
from deap_er.private.programming.tape_lower import lower_tree
from deap_er.private.various.rng import rng
from deap_er.private.various.semantic_neighbors import SemanticMetric, semantic_nearest

__all__: list[str] = ["cx_one_point_semantic"]


def _subtree_tape_rows(
    indices: Sequence[int],
    individual: GPIndividual,
    prim_set: PrimitiveSetTyped,
    matrix: numpy.ndarray,
) -> numpy.ndarray:
    """Lower and score each subtree index against ``matrix``.

    Args:
        indices: Node indices whose subtrees should be evaluated.
        individual: Tree that owns the subtrees.
        prim_set: Primitive set used to lower each subtree.
        matrix: Packed ``(n_rows, n_columns)`` input matrix.

    Returns:
        Semantic rows of shape ``(len(indices), n_rows)``.
    """
    tapes = []
    for index in indices:
        subtree = PrimitiveTree(individual[individual.search_subtree(index)])
        tapes.append(lower_tree(subtree, prim_set))
    return interpret_tapes(tapes, matrix)


def cx_one_point_semantic(
    ind1: GPIndividual,
    ind2: GPIndividual,
    prim_set: PrimitiveSetTyped,
    matrix: numpy.ndarray,
    *,
    valid: numpy.ndarray | None = None,
    metric: SemanticMetric = "euclidean",
) -> GPMates:
    """Exchange subtrees at a semantically close type-matched pair.

    One shared return type is chosen at random. ``ind1`` donates a
    random crossover point of that type. ``ind2`` donates the
    type-matched node whose ``interpret_tape`` row is nearest to the
    anchor subtree under ``metric``. When every distance is infinite,
    ``ind2``'s partner is chosen uniformly among its type-matched
    candidates. Type-matched one-point (:func:`~deap_er.gp.cx_one_point`)
    remains the default toolbox mate; this operator is for columnar runs
    that already score ``interpret_tapes``. Geometric semantic variation
    is :func:`~deap_er.gp.cx_semantic`.

    Args:
        ind1: First individual to mate.
        ind2: Second individual to mate.
        prim_set: Primitive set that can lower every subtree primitive.
        matrix: Packed ``(n_rows, n_columns)`` matrix for
            :func:`~deap_er.gp.interpret_tapes`.
        valid: Optional per-row warmup mask of length ``n_rows``.
        metric: ``euclidean`` or ``cosine`` distance on finite rows.

    Returns:
        The two individuals after subtree exchange.

    Raises:
        ValueError: If a subtree cannot be lowered to a tape.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2

    types1, types2, common_types = _common_type_candidates(ind1, ind2)
    if len(common_types) == 0:
        return ind1, ind2

    type_ = rng.choice(list(common_types))
    cands1 = types1[type_]
    cands2 = types2[type_]
    index1 = int(rng.choice(cands1))

    rows1 = _subtree_tape_rows([index1], ind1, prim_set, matrix)
    rows2 = _subtree_tape_rows(cands2, ind2, prim_set, matrix)
    nearest = semantic_nearest(rows1[0], rows2, k=1, metric=metric, valid=valid)
    index2 = int(rng.choice(cands2)) if nearest.size == 0 else cands2[int(nearest[0])]

    _swap_at(ind1, ind2, index1, index2)
    return ind1, ind2

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
import random
from collections import defaultdict
from functools import partial
from operator import eq, lt

from .dtypes import *

__all__ = ["cx_one_point", "cx_one_point_leaf_biased"]


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

    types1 = defaultdict(list)
    types2 = defaultdict(list)
    if ind1.root.ret is object:
        types1[object] = list(range(1, len(ind1)))
        types2[object] = list(range(1, len(ind2)))
        common_types = [object]
    else:
        for idx, node in enumerate(ind1[1:], 1):
            types1[node.ret].append(idx)
        for idx, node in enumerate(ind2[1:], 1):
            types2[node.ret].append(idx)
        common_types = set(types1.keys()).intersection(set(types2.keys()))

    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])
        slice1 = ind1.search_subtree(index1)
        slice2 = ind2.search_subtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

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
    arity_op1 = terminal_op if random.random() < term_prob else primitive_op
    arity_op2 = terminal_op if random.random() < term_prob else primitive_op

    types1 = defaultdict(list)
    types2 = defaultdict(list)

    for idx, node in enumerate(ind1[1:], 1):
        if arity_op1(node.arity):
            types1[node.ret].append(idx)

    for idx, node in enumerate(ind2[1:], 1):
        if arity_op2(node.arity):
            types2[node.ret].append(idx)

    common_types = set(types1.keys()).intersection(set(types2.keys()))

    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])
        slice1 = ind1.search_subtree(index1)
        slice2 = ind2.search_subtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2

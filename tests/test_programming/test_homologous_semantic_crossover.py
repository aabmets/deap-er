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
from typing import Any

import numpy
from deap_er import gp, tools
from deap_er.private.programming.crossover import common_type_candidates, swap_at
from deap_er.private.programming.cx_semantic_one_point import subtree_tape_rows

COLUMNS = ["first", "second", "third"]


def _kit(window_name: str = "HOMOLOGOUS_SEMANTIC"):
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    gp.add_window_primitives(pset)
    gp.add_window_ephemeral(pset, window_name, 1, 5)
    return pset


def _matrix(size: int = 8) -> numpy.ndarray:
    generator = numpy.random.default_rng(19)
    columns = [
        generator.normal(size=size),
        generator.normal(size=size) * 2.0,
        numpy.abs(generator.normal(size=size)),
    ]
    return numpy.ascontiguousarray(numpy.stack(columns, axis=1))


def _assert_well_typed(tree: gp.PrimitiveTree) -> None:
    stack: list[type] = []
    for node in reversed(list(tree)):
        if node.arity == 0:
            stack.append(node.ret)
            continue
        args = [stack.pop() for _ in range(node.arity)]
        assert args == list(node.args)
        stack.append(node.ret)


def test_semantic_picks_semantically_nearest_partner():

    pset = _kit()
    matrix = _matrix()
    matched = False
    for seed in range(100):
        tools.rng.seed(seed)
        ind1: Any = gp.PrimitiveTree.from_string("vadd(second, third)", pset)
        ind2: Any = gp.PrimitiveTree.from_string("vadd(first, second)", pset)
        types1, types2, common_types = common_type_candidates(ind1, ind2)
        if len(common_types) == 0:
            continue
        type_ = tools.rng.choice(list(common_types))
        cands1 = types1[type_]
        cands2 = types2[type_]
        index1 = int(tools.rng.choice(cands1))
        rows1 = subtree_tape_rows([index1], ind1, pset, matrix)
        rows2 = subtree_tape_rows(cands2, ind2, pset, matrix)
        nearest = gp.semantic_nearest(rows1[0], rows2, k=1, metric="euclidean")
        index2 = int(tools.rng.choice(cands2)) if nearest.size == 0 else cands2[int(nearest[0])]

        expected_first: Any = gp.PrimitiveTree.from_string("vadd(second, third)", pset)
        expected_second: Any = gp.PrimitiveTree.from_string("vadd(first, second)", pset)
        swap_at(expected_first, expected_second, index1, index2)

        tools.rng.seed(seed)
        actual_first: Any = gp.PrimitiveTree.from_string("vadd(second, third)", pset)
        actual_second: Any = gp.PrimitiveTree.from_string("vadd(first, second)", pset)
        gp.cx_one_point_semantic(actual_first, actual_second, pset, matrix)

        if str(actual_first) == str(expected_first) and str(actual_second) == str(expected_second):
            matched = True
            break
    assert matched


def test_semantic_falls_back_when_all_distances_infinite():
    pset = _kit()
    matrix = _matrix()
    valid = numpy.zeros(matrix.shape[0], dtype=bool)
    changed = False
    for seed in range(30):
        tools.rng.seed(seed)
        ind1: Any = gp.PrimitiveTree.from_string("vadd(first, second)", pset)
        ind2: Any = gp.PrimitiveTree.from_string("vadd(third, third)", pset)
        before_first = str(ind1)
        before_second = str(ind2)
        gp.cx_one_point_semantic(ind1, ind2, pset, matrix, valid=valid)
        if str(ind1) != before_first or str(ind2) != before_second:
            changed = True
            _assert_well_typed(ind1)
            _assert_well_typed(ind2)
            break
    assert changed


def test_semantic_short_trees_unchanged():
    pset = _kit()
    matrix = _matrix()
    first: Any = gp.PrimitiveTree.from_string("first", pset)
    second: Any = gp.PrimitiveTree.from_string("second", pset)
    before_first = list(first)
    before_second = list(second)
    gp.cx_one_point_semantic(first, second, pset, matrix)
    assert list(first) == before_first
    assert list(second) == before_second


def test_semantic_preserves_types():
    pset = _kit()
    matrix = _matrix()
    for seed in range(20):
        tools.rng.seed(seed)
        first: Any = gp.PrimitiveTree(gp.gen_half_and_half(pset, 2, 3))
        second: Any = gp.PrimitiveTree(gp.gen_half_and_half(pset, 2, 3))
        gp.cx_one_point_semantic(first, second, pset, matrix)
        _assert_well_typed(first)
        _assert_well_typed(second)


def test_homologous_and_semantic_neverswap_at_root():
    pset = _kit()
    matrix = _matrix()
    first: Any = gp.PrimitiveTree.from_string("vadd(vadd(first, second), third)", pset)
    second: Any = gp.PrimitiveTree.from_string("vadd(vadd(third, second), first)", pset)
    first_root = first[0]
    second_root = second[0]
    for seed in range(50):
        tools.rng.seed(seed)
        left: Any = gp.PrimitiveTree.from_string(str(first), pset)
        right: Any = gp.PrimitiveTree.from_string(str(second), pset)
        gp.cx_homologous(left, right)
        assert left[0] == first_root
        assert right[0] == second_root
        tools.rng.seed(seed)
        left: Any = gp.PrimitiveTree.from_string(str(first), pset)
        right: Any = gp.PrimitiveTree.from_string(str(second), pset)
        gp.cx_one_point_semantic(left, right, pset, matrix)
        assert left[0] == first_root
        assert right[0] == second_root


def test_semantic_all_inf_fallback_matches_random_partner_oracle():

    pset = _kit()
    matrix = _matrix()
    valid = numpy.zeros(matrix.shape[0], dtype=bool)
    tools.rng.seed(9)
    ind1: Any = gp.PrimitiveTree.from_string("vadd(first, second)", pset)
    ind2: Any = gp.PrimitiveTree.from_string("vadd(third, third)", pset)
    types1, types2, common_types = common_type_candidates(ind1, ind2)
    type_ = tools.rng.choice(list(common_types))
    cands1 = types1[type_]
    cands2 = types2[type_]
    index1 = int(tools.rng.choice(cands1))
    rows1 = subtree_tape_rows([index1], ind1, pset, matrix)
    rows2 = subtree_tape_rows(cands2, ind2, pset, matrix)
    nearest = gp.semantic_nearest(rows1[0], rows2, k=1, metric="euclidean", valid=valid)
    assert nearest.size == 0
    index2 = int(tools.rng.choice(cands2))

    expected_first: Any = gp.PrimitiveTree.from_string("vadd(first, second)", pset)
    expected_second: Any = gp.PrimitiveTree.from_string("vadd(third, third)", pset)
    swap_at(expected_first, expected_second, index1, index2)

    tools.rng.seed(9)
    actual_first: Any = gp.PrimitiveTree.from_string("vadd(first, second)", pset)
    actual_second: Any = gp.PrimitiveTree.from_string("vadd(third, third)", pset)
    gp.cx_one_point_semantic(actual_first, actual_second, pset, matrix, valid=valid)

    assert str(actual_first) == str(expected_first)
    assert str(actual_second) == str(expected_second)

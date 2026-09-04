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
from deap_er import tools


class _ESList(list[Any]):
    def __init__(self, genes: list[float], strategy: list[float]) -> None:
        super().__init__(genes)
        self.strategy = list(strategy)


def _es(genes: list[float], strategy: list[float]) -> Any:
    return _ESList(genes, strategy)


def test_one_point_swaps_tails():
    tools.seed(1)
    left: Any = [0, 1, 2, 3, 4]
    right: Any = [9, 8, 7, 6, 5]
    first, second = tools.cx_one_point(left, right)
    assert len(first) == len(second) == 5
    assert sorted(list(first) + list(second)) == sorted([0, 1, 2, 3, 4, 9, 8, 7, 6, 5])


def test_messy_one_point_may_change_length():
    tools.seed(2)
    left: Any = [0, 1, 2, 3]
    right: Any = [9, 8, 7, 6, 5]
    first, second = tools.cx_messy_one_point(left, right)
    assert len(first) + len(second) == 9


def test_two_point_variants():
    tools.seed(3)
    left: Any = [0, 1, 2, 3, 4]
    right: Any = [9, 8, 7, 6, 5]
    first, second = tools.cx_two_point(left, right)
    assert len(first) == len(second) == 5

    tools.seed(3)
    arr1: Any = numpy.array([0, 1, 2, 3, 4])
    arr2: Any = numpy.array([9, 8, 7, 6, 5])
    copy1, copy2 = tools.cx_two_point_copy(arr1, arr2)
    assert list(copy1) == list(first)
    assert list(copy2) == list(second)


def test_es_two_point_also_swaps_strategy():
    first = _es([0.0, 1.0, 2.0, 3.0], [0.1, 0.2, 0.3, 0.4])
    second = _es([9.0, 8.0, 7.0, 6.0], [1.1, 1.2, 1.3, 1.4])
    tools.seed(4)
    tools.cx_es_two_point(first, second)
    assert len(first) == len(first.strategy) == 4
    assert len(second) == len(second.strategy) == 4

    tools.seed(4)
    copy1 = _es([0.0, 1.0, 2.0, 3.0], [0.1, 0.2, 0.3, 0.4])
    copy2 = _es([9.0, 8.0, 7.0, 6.0], [1.1, 1.2, 1.3, 1.4])
    tools.cx_es_two_point_copy(copy1, copy2)
    assert len(copy1) == len(copy1.strategy) == 4

    tools.seed(4)
    arr1: Any = numpy.array([0.0, 1.0, 2.0, 3.0])
    arr2: Any = numpy.array([9.0, 8.0, 7.0, 6.0])
    tools.cx_two_point_copy(arr1, arr2)
    assert len(arr1) == 4


def test_partially_matched_and_uniform_pmx_keep_permutations():
    tools.seed(5)
    pmx_left: Any = [0, 1, 2, 3, 4, 5]
    pmx_right: Any = [5, 4, 3, 2, 1, 0]
    first, second = tools.cx_partially_matched(pmx_left, pmx_right)
    assert sorted(first) == list(range(6))
    assert sorted(second) == list(range(6))

    tools.seed(6)
    upmx_left: Any = [0, 1, 2, 3, 4, 5]
    upmx_right: Any = [5, 4, 3, 2, 1, 0]
    first, second = tools.cx_uniform_partially_matched(upmx_left, upmx_right, 0.5)
    assert sorted(first) == list(range(6))
    assert sorted(second) == list(range(6))


def test_blend_and_simulated_binary():
    tools.seed(7)
    left: Any = [0.0, 1.0, 2.0]
    right: Any = [1.0, 2.0, 3.0]
    first, second = tools.cx_blend(left, right, 0.5)
    assert len(first) == len(second) == 3

    first = _es([0.0, 1.0], [0.2, 0.3])
    second = _es([2.0, 3.0], [0.4, 0.5])
    tools.seed(8)
    tools.cx_es_blend(first, second, 0.3)
    assert len(first.strategy) == 2

    tools.seed(9)
    sbx_left: Any = [0.0, 1.0]
    sbx_right: Any = [1.0, 0.0]
    first, second = tools.cx_simulated_binary(sbx_left, sbx_right, eta=2.0)
    assert len(first) == 2


def test_uniform_and_ordered():
    tools.seed(10)
    left: Any = [0, 1, 2, 3]
    right: Any = [9, 8, 7, 6]
    first, second = tools.cx_uniform(left, right, 0.5)
    assert len(first) == len(second) == 4

    tools.seed(11)
    order_left: Any = [0, 1, 2, 3, 4]
    order_right: Any = [4, 3, 2, 1, 0]
    first, second = tools.cx_ordered(order_left, order_right)
    assert sorted(first) == list(range(5))
    assert sorted(second) == list(range(5))

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
import array
from typing import Any

import numpy
from deap_er import tools
from deap_er.private.operators.cx_point import slicer


class _ESList(list[Any]):
    def __init__(self, genes: list[float], strategy: list[float]) -> None:
        super().__init__(genes)
        self.strategy = list(strategy)


def _es(genes: list[float], strategy: list[float]) -> Any:
    return _ESList(genes, strategy)


def test_cx_one_point_length_one_is_noop():
    left: Any = [0]
    right: Any = [1]
    assert tools.cx_one_point(left, right) == ([0], [1])


def test_cx_two_point_length_one_is_noop():
    left: Any = [0]
    right: Any = [1]
    assert tools.cx_two_point(left, right) == ([0], [1])


def test_one_point_swaps_tails():
    tools.rng.seed(1)
    left: Any = [0, 1, 2, 3, 4]
    right: Any = [9, 8, 7, 6, 5]
    first, second = tools.cx_one_point(left, right)
    assert len(first) == len(second) == 5
    assert sorted(list(first) + list(second)) == sorted([0, 1, 2, 3, 4, 9, 8, 7, 6, 5])


def test_slicer_swaps_numpy_without_aliasing():
    left: Any = numpy.array([0, 1, 2, 3, 4])
    right: Any = numpy.array([9, 8, 7, 6, 5])

    first, second = slicer(left.copy(), right.copy(), 2)
    assert list(first) == [0, 1, 7, 6, 5]
    assert list(second) == [9, 8, 2, 3, 4]


def test_messy_one_point_may_change_length():
    tools.rng.seed(2)
    left: Any = [0, 1, 2, 3]
    right: Any = [9, 8, 7, 6, 5]
    first, second = tools.cx_messy_one_point(left, right)
    assert len(first) + len(second) == 9


def test_messy_one_point_swaps_independent_tails():
    left_genes = [0, 1, 2, 3]
    right_genes = [9, 8, 7, 6]
    saw_length_change = False
    for seed in range(80):
        tools.rng.seed(seed)
        cut1 = tools.rng.randint(0, len(left_genes))
        cut2 = tools.rng.randint(0, len(right_genes))
        tools.rng.seed(seed)
        parent1: Any = list(left_genes)
        parent2: Any = list(right_genes)
        first, second = tools.cx_messy_one_point(parent1, parent2)
        assert list(first) == left_genes[:cut1] + right_genes[cut2:]
        assert list(second) == right_genes[:cut2] + left_genes[cut1:]
        if len(first) != len(left_genes) or len(second) != len(right_genes):
            saw_length_change = True
    assert saw_length_change


def test_two_point_variants():
    tools.rng.seed(3)
    left: Any = [0, 1, 2, 3, 4]
    right: Any = [9, 8, 7, 6, 5]
    first, second = tools.cx_two_point(left, right)
    assert len(first) == len(second) == 5

    tools.rng.seed(3)
    arr1: Any = numpy.array([0, 1, 2, 3, 4])
    arr2: Any = numpy.array([9, 8, 7, 6, 5])
    copy1, copy2 = tools.cx_two_point_copy(arr1, arr2)
    assert list(copy1) == list(first)
    assert list(copy2) == list(second)


def test_cx_two_point_swaps_array_module_slices():
    tools.rng.seed(3)
    list_left: Any = [0, 1, 2, 3, 4]
    list_right: Any = [9, 8, 7, 6, 5]
    list1, list2 = tools.cx_two_point(list_left, list_right)
    tools.rng.seed(3)
    first: Any = array.array("b", [0, 1, 2, 3, 4])
    second: Any = array.array("b", [9, 8, 7, 6, 5])
    first, second = tools.cx_two_point(first, second)
    assert isinstance(first, array.array)
    assert isinstance(second, array.array)
    assert list(first) == list1
    assert list(second) == list2


def test_cx_messy_one_point_swaps_array_module_slices():
    tools.rng.seed(2)
    first: Any = array.array("b", [0, 1, 2, 3])
    second: Any = array.array("b", [9, 8, 7, 6, 5])
    first, second = tools.cx_messy_one_point(first, second)
    assert isinstance(first, array.array)
    assert isinstance(second, array.array)
    assert len(first) + len(second) == 9


def test_es_two_point_also_swaps_strategy():
    first = _es([0.0, 1.0, 2.0, 3.0], [0.1, 0.2, 0.3, 0.4])
    second = _es([9.0, 8.0, 7.0, 6.0], [1.1, 1.2, 1.3, 1.4])
    tools.rng.seed(4)
    tools.cx_es_two_point(first, second)
    assert len(first) == len(first.strategy) == 4
    assert len(second) == len(second.strategy) == 4

    tools.rng.seed(4)
    copy1 = _es([0.0, 1.0, 2.0, 3.0], [0.1, 0.2, 0.3, 0.4])
    copy2 = _es([9.0, 8.0, 7.0, 6.0], [1.1, 1.2, 1.3, 1.4])
    tools.cx_es_two_point_copy(copy1, copy2)
    assert len(copy1) == len(copy1.strategy) == 4

    tools.rng.seed(4)
    arr1: Any = numpy.array([0.0, 1.0, 2.0, 3.0])
    arr2: Any = numpy.array([9.0, 8.0, 7.0, 6.0])
    tools.cx_two_point_copy(arr1, arr2)
    assert len(arr1) == 4


def test_es_two_point_copy_swaps_numpy_strategy():
    class _EsArray(numpy.ndarray):
        strategy: Any

    def make(genes: list[float], strategy: list[float]) -> Any:
        ind = numpy.array(genes, dtype=float).view(_EsArray)
        ind.strategy = numpy.array(strategy, dtype=float)
        return ind

    first = make([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.10, 0.11, 0.12, 0.13, 0.14, 0.15])
    second = make([9.0, 8.0, 7.0, 6.0, 5.0, 4.0], [0.90, 0.91, 0.92, 0.93, 0.94, 0.95])
    orig_first_sigma = first.strategy.copy()
    orig_second_sigma = second.strategy.copy()
    tools.rng.seed(7)
    tools.cx_es_two_point_copy(first, second)
    assert not numpy.array_equal(first.strategy, orig_first_sigma) or not numpy.array_equal(
        second.strategy, orig_second_sigma
    )
    combined = sorted(first.strategy.tolist() + second.strategy.tolist())
    assert combined == sorted(orig_first_sigma.tolist() + orig_second_sigma.tolist())

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
from deap_er.private.operators.cx_point import slicer


class _ESList(list[Any]):
    def __init__(self, genes: list[float], strategy: list[float]) -> None:
        super().__init__(genes)
        self.strategy = list(strategy)


def _es(genes: list[float], strategy: list[float]) -> Any:
    return _ESList(genes, strategy)


def test_one_point_swaps_tails():
    tools.rng.seed(1)
    left: Any = [0, 1, 2, 3, 4]
    right: Any = [9, 8, 7, 6, 5]
    first, second = tools.cx_one_point(left, right)
    assert len(first) == len(second) == 5
    assert sorted(list(first) + list(second)) == sorted([0, 1, 2, 3, 4, 9, 8, 7, 6, 5])


def test_slicer_and_uniform_swap_numpy_without_aliasing():
    left: Any = numpy.array([0, 1, 2, 3, 4])
    right: Any = numpy.array([9, 8, 7, 6, 5])

    first, second = slicer(left.copy(), right.copy(), 2)
    assert list(first) == [0, 1, 7, 6, 5]
    assert list(second) == [9, 8, 2, 3, 4]

    tools.rng.seed(0)
    uni_left: Any = numpy.array([0, 1, 2, 3, 4])
    uni_right: Any = numpy.array([9, 8, 7, 6, 5])
    tools.cx_uniform(uni_left, uni_right, 1.0)
    assert list(uni_left) == [9, 8, 7, 6, 5]
    assert list(uni_right) == [0, 1, 2, 3, 4]


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


def test_partially_matched_and_uniform_pmx_keep_permutations():
    tools.rng.seed(5)
    pmx_left: Any = [0, 1, 2, 3, 4, 5]
    pmx_right: Any = [5, 4, 3, 2, 1, 0]
    first, second = tools.cx_partially_matched(pmx_left, pmx_right)
    assert sorted(first) == list(range(6))
    assert sorted(second) == list(range(6))

    tools.rng.seed(6)
    upmx_left: Any = [0, 1, 2, 3, 4, 5]
    upmx_right: Any = [5, 4, 3, 2, 1, 0]
    first, second = tools.cx_uniform_partially_matched(upmx_left, upmx_right, 0.5)
    assert sorted(first) == list(range(6))
    assert sorted(second) == list(range(6))


def test_simulated_binary_bounded_second_child_uses_plus_side():
    eta = 2.0
    x1, x2, low, up, rand = 0.2, 0.6, 0.0, 1.0, 0.3

    def beta_q(diff: float) -> float:
        beta = 1.0 + (2.0 * diff / (x2 - x1))
        alpha = 2.0 - beta ** -(eta + 1)
        if rand <= 1.0 / alpha:
            return float((rand * alpha) ** (1.0 / (eta + 1)))
        return float((1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1)))

    expected_c1 = 0.5 * (x1 + x2 - beta_q(x1 - low) * (x2 - x1))
    expected_c2 = 0.5 * (x1 + x2 + beta_q(up - x2) * (x2 - x1))
    midpoint = (x1 + x2) / 2.0
    assert expected_c1 < midpoint < expected_c2

    saw_upper_child = False
    crossed = 0
    for seed in range(300):
        tools.rng.seed(seed)
        first: Any = [x1]
        second: Any = [x2]
        tools.cx_simulated_binary_bounded(first, second, eta, low, up)
        if {first[0], second[0]} == {x1, x2}:
            continue
        crossed += 1
        if first[0] > midpoint or second[0] > midpoint:
            saw_upper_child = True
    assert crossed > 0
    assert saw_upper_child


def test_blend_and_simulated_binary():
    tools.rng.seed(7)
    left: Any = [0.0, 1.0, 2.0]
    right: Any = [1.0, 2.0, 3.0]
    first, second = tools.cx_blend(left, right, 0.5)
    assert len(first) == len(second) == 3

    first = _es([0.0, 1.0], [0.2, 0.3])
    second = _es([2.0, 3.0], [0.4, 0.5])
    tools.rng.seed(8)
    tools.cx_es_blend(first, second, 0.3)
    assert len(first.strategy) == 2

    tools.rng.seed(9)
    sbx_left: Any = [0.0, 1.0]
    sbx_right: Any = [1.0, 0.0]
    first, second = tools.cx_simulated_binary(sbx_left, sbx_right, eta=2.0)
    assert len(first) == 2


def test_uniform_and_ordered():
    tools.rng.seed(10)
    left: Any = [0, 1, 2, 3]
    right: Any = [9, 8, 7, 6]
    first, second = tools.cx_uniform(left, right, 0.5)
    assert len(first) == len(second) == 4

    tools.rng.seed(11)
    order_left: Any = [0, 1, 2, 3, 4]
    order_right: Any = [4, 3, 2, 1, 0]
    first, second = tools.cx_ordered(order_left, order_right)
    assert sorted(first) == list(range(5))
    assert sorted(second) == list(range(5))


def test_blend_bounded_and_sbx_out_of_box_stay_finite():
    tools.rng.seed(12)
    blend_left: Any = [0.0, 1.0]
    blend_right: Any = [1.0, 0.0]
    first, second = tools.cx_blend_bounded(blend_left, blend_right, 0.5, 0.0, 1.0)
    assert all(0.0 <= gene <= 1.0 for gene in list(first) + list(second))

    for seed in range(20):
        tools.rng.seed(seed)
        sbx_left: Any = [10.0]
        sbx_right: Any = [-5.0]
        first, second = tools.cx_simulated_binary_bounded(sbx_left, sbx_right, 2.0, 0.0, 1.0)
        kids = list(first) + list(second)
        assert all(numpy.isfinite(gene) for gene in kids)
        assert all(not isinstance(gene, complex) for gene in kids)
        for gene in kids:
            if gene not in (10.0, -5.0):
                assert 0.0 <= gene <= 1.0


def test_pmx_and_ordered_accept_letter_permutations():
    letters = list("abc")
    tools.rng.seed(14)
    pmx_left: Any = list(letters)
    pmx_right: Any = list("cba")
    first, second = tools.cx_partially_matched(pmx_left, pmx_right)
    assert sorted(first) == letters and sorted(second) == letters

    tools.rng.seed(15)
    upmx_left: Any = list(letters)
    upmx_right: Any = list("cba")
    first, second = tools.cx_uniform_partially_matched(upmx_left, upmx_right, 0.5)
    assert sorted(first) == letters and sorted(second) == letters

    tools.rng.seed(16)
    order_left: Any = list(letters)
    order_right: Any = list("cba")
    first, second = tools.cx_ordered(order_left, order_right)
    assert sorted(first) == letters and sorted(second) == letters

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


def test_uniform_swaps_numpy_without_aliasing():
    tools.rng.seed(0)
    uni_left: Any = numpy.array([0, 1, 2, 3, 4])
    uni_right: Any = numpy.array([9, 8, 7, 6, 5])
    tools.cx_uniform(uni_left, uni_right, 1.0)
    assert list(uni_left) == [9, 8, 7, 6, 5]
    assert list(uni_right) == [0, 1, 2, 3, 4]


def test_uniform_keeps_length():
    tools.rng.seed(10)
    left: Any = [0, 1, 2, 3]
    right: Any = [9, 8, 7, 6]
    first, second = tools.cx_uniform(left, right, 0.5)
    assert len(first) == len(second) == 4


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

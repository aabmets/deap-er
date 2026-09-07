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

from deap_er import tools


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


def test_cx_ordered_length_one_is_noop():
    left: Any = [0]
    right: Any = [0]
    assert tools.cx_ordered(left, right) == ([0], [0])
    empty_left: Any = []
    empty_right: Any = []
    assert tools.cx_ordered(empty_left, empty_right) == ([], [])


def test_ordered_keeps_permutations():
    tools.rng.seed(11)
    order_left: Any = [0, 1, 2, 3, 4]
    order_right: Any = [4, 3, 2, 1, 0]
    first, second = tools.cx_ordered(order_left, order_right)
    assert sorted(first) == list(range(5))
    assert sorted(second) == list(range(5))


def test_pmx_and_ordered_accept_letter_permutations():
    letters = list("abc")
    tools.rng.seed(14)
    pmx_left: Any = list(letters)
    pmx_right: Any = list("cba")
    first, second = tools.cx_partially_matched(pmx_left, pmx_right)
    assert sorted(first) == letters
    assert sorted(second) == letters

    tools.rng.seed(15)
    upmx_left: Any = list(letters)
    upmx_right: Any = list("cba")
    first, second = tools.cx_uniform_partially_matched(upmx_left, upmx_right, 0.5)
    assert sorted(first) == letters
    assert sorted(second) == letters

    tools.rng.seed(16)
    order_left: Any = list(letters)
    order_right: Any = list("cba")
    first, second = tools.cx_ordered(order_left, order_right)
    assert sorted(first) == letters
    assert sorted(second) == letters

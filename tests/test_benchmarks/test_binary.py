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


def _bits(pattern: list[int], length: int) -> Any:
    return (pattern * ((length // len(pattern)) + 1))[:length]


def test_royal_road_1_scores_complete_blocks():
    ones: Any = [1, 1, 1, 1, 1, 1, 1, 1]
    zeros: Any = [0] * 8
    mixed: Any = [1, 1, 0, 0, 1, 1, 1, 1]
    assert tools.bm_royal_road_1(ones, 4) == (8,)
    assert tools.bm_royal_road_1(zeros, 4) == (0,)
    assert tools.bm_royal_road_1(mixed, 4) == (4,)


def test_royal_road_2_sums_successive_orders():
    ones: Any = [1] * 16
    zeros: Any = [0] * 16
    assert tools.bm_royal_road_2(ones, 4)[0] == (
        tools.bm_royal_road_1(ones, 4)[0] + tools.bm_royal_road_1(ones, 8)[0]
    )
    assert tools.bm_royal_road_2(zeros, 4) == (0,)


def test_chuang_f1_uses_trap_or_inverse_from_last_bit():
    zeros: Any = [0] * 41
    ones: Any = [1] * 41
    mixed: Any = [1, 1, 1, 1] * 10 + [0]

    assert tools.bm_chuang_f1(zeros) == (40,)
    assert tools.bm_chuang_f1(ones) == (40,)
    assert tools.bm_chuang_f1(mixed)[0] < 40


def test_chuang_f2_covers_all_suffix_combinations():
    scores = []
    for suffix in ([0, 0], [0, 1], [1, 0], [1, 1]):
        individual: Any = [1] * 39 + suffix
        value = tools.bm_chuang_f2(individual)
        assert len(value) == 1
        scores.append(value[0])

    assert len(set(scores)) > 1


def test_chuang_f3_wraps_the_first_pair_when_the_last_bit_is_one():
    zeros: Any = [0] * 41
    ones: Any = [1] * 41
    mixed: Any = [1, 0] * 20 + [1]

    assert tools.bm_chuang_f3(zeros)[0] > 0
    assert tools.bm_chuang_f3(ones)[0] > 0
    assert tools.bm_chuang_f3(mixed)[0] >= 0
    assert tools.bm_chuang_f3(_bits([0, 1], 41))[0] >= 0

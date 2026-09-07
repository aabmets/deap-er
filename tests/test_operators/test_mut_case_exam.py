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
import numpy
import pytest
from deap_er import tools


def test_mut_case_ranges_zero_prob_is_noop():
    ranges = [(1, 4), (4, 8)]
    (out,) = tools.mut_case_ranges(ranges, length=8, mut_prob=0.0)
    assert out == [(1, 4), (4, 8)]
    assert out is ranges


def test_mut_case_ranges_stays_inside_bounds():
    tools.rng.seed(7)
    ranges = [(2, 5)]
    (out,) = tools.mut_case_ranges(ranges, length=6, mut_prob=1.0)
    start, stop = out[0]
    assert 0 <= start <= stop <= 6


def test_mut_case_ranges_empty_list_and_negative_length():
    ranges: list[tuple[int, int]] = []
    assert tools.mut_case_ranges(ranges, length=4, mut_prob=1.0) == ([],)
    with pytest.raises(ValueError, match="non-negative"):
        tools.mut_case_ranges([(0, 1)], length=-1, mut_prob=1.0)


def test_mut_case_mask_zero_prob_is_noop():
    mask = numpy.array([True, True, False, False], dtype=bool)
    (out,) = tools.mut_case_mask(mask, mut_prob=0.0)
    assert numpy.array_equal(out, [True, True, False, False])
    assert out is mask


def test_mut_case_mask_flips_whole_runs():
    tools.rng.seed(3)
    mask = numpy.array([True, True, False, False, True], dtype=bool)
    (out,) = tools.mut_case_mask(mask, mut_prob=1.0)
    assert numpy.array_equal(out, [False, False, True, True, False])
    assert out is mask


def test_mut_case_mask_empty_is_noop():
    mask = numpy.array([], dtype=bool)
    (out,) = tools.mut_case_mask(mask, mut_prob=1.0)
    assert out.size == 0

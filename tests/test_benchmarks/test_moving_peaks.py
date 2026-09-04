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

from deap_er.benchmarks.moving_peaks import MovingPeaks


def test_offline_error_before_first_evaluation():
    landscape = MovingPeaks(dimensions=2)

    assert landscape.offline_error == 0.0


def test_offline_error_after_evaluation():
    landscape = MovingPeaks(dimensions=2)

    landscape([50.0, 50.0])

    assert landscape.nevals == 1
    assert landscape.offline_error >= 0.0


def test_fluctuating_peak_count_is_accepted():
    landscape = MovingPeaks(dimensions=2, npeaks=[3, 5, 8], change_severity=1.0)

    assert landscape.min_peaks == 3
    assert landscape.max_peaks == 8
    assert landscape.number_severity == 1.0
    assert len(landscape.peaks_function) == 5


def test_fluctuating_peak_count_defaults_to_no_change_severity():
    landscape = MovingPeaks(dimensions=2, npeaks=[3, 5, 8])

    assert landscape.number_severity == 0.0


def test_change_peaks_respects_peak_count_bounds():
    random.seed(17)
    landscape = MovingPeaks(dimensions=2, npeaks=[3, 5, 8], change_severity=1.0)

    for _ in range(10):
        landscape.change_peaks()
        assert 3 <= len(landscape.peaks_function) <= 8
        assert len(landscape.peaks_position) == len(landscape.peaks_function)
        assert len(landscape.peaks_height) == len(landscape.peaks_function)
        assert len(landscape.peaks_width) == len(landscape.peaks_function)

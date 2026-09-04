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

import pytest
from deap_er import tools
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
    tools.seed(17)
    landscape = MovingPeaks(dimensions=2, npeaks=[3, 5, 8], change_severity=1.0)

    for _ in range(10):
        landscape.change_peaks()
        assert 3 <= len(landscape.peaks_function) <= 8
        assert len(landscape.peaks_position) == len(landscape.peaks_function)
        assert len(landscape.peaks_height) == len(landscape.peaks_function)
        assert len(landscape.peaks_width) == len(landscape.peaks_function)


def test_change_peaks_is_stable_with_a_fluctuating_count():
    # Characterization: pins the interleaving of the peak-count change and the
    # per-peak position, height, and width updates.
    tools.seed(77)
    landscape = MovingPeaks(dimensions=2, npeaks=[2, 3, 5], change_severity=1.0)

    landscape.change_peaks()

    assert len(landscape.peaks_function) == 2
    flat_positions = [coord for position in landscape.peaks_position for coord in position]
    assert flat_positions == pytest.approx(
        [80.7428994728303, 9.870377604514095, 76.38848077545254, 59.544151415190086]
    )
    assert landscape.peaks_height == pytest.approx([50.800338213253816, 59.706739130737255])
    assert landscape.peaks_width == pytest.approx([0.11793668124545155, 0.11189864736162168])


def test_change_peaks_is_stable_with_a_fixed_count():
    tools.seed(5)
    landscape = MovingPeaks(dimensions=2)

    landscape.change_peaks()

    assert len(landscape.peaks_function) == 5
    assert landscape.peaks_height == pytest.approx(
        [
            49.07474653839666,
            39.459476387341326,
            47.88750297144452,
            49.643784785486034,
            57.39053280972766,
        ]
    )
    assert landscape.peaks_position[0] == pytest.approx([65.47327583938005, 22.479355750749242])

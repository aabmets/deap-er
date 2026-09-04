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

import pytest
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


def test_change_peaks_is_stable_with_a_fluctuating_count():
    # Characterization: pins the interleaving of the peak-count change and the
    # per-peak position, height, and width updates.
    random.seed(77)
    landscape = MovingPeaks(dimensions=2, npeaks=[2, 3, 5], change_severity=1.0)

    landscape.change_peaks()

    assert len(landscape.peaks_function) == 2
    flat_positions = [coord for position in landscape.peaks_position for coord in position]
    assert flat_positions == pytest.approx(
        [15.234792408525824, 62.33901224723937, 49.658985446975244, 84.19668008266241]
    )
    assert landscape.peaks_height == pytest.approx([56.942552763348274, 44.840135030472624])
    assert landscape.peaks_width == pytest.approx([0.09985828374029866, 0.09755426652252269])


def test_change_peaks_is_stable_with_a_fixed_count():
    random.seed(5)
    landscape = MovingPeaks(dimensions=2)

    landscape.change_peaks()

    assert len(landscape.peaks_function) == 5
    assert landscape.peaks_height == pytest.approx(
        [
            51.11696123992914,
            64.17678979838504,
            64.11815130548122,
            39.782571152097994,
            43.35067780087705,
        ]
    )
    assert landscape.peaks_position[0] == pytest.approx([90.70539297560782, 10.532337620054733])

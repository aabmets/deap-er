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
from deap_er.benchmarks import MovingPeaks


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
    tools.rng.seed(17)
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
    tools.rng.seed(77)
    landscape = MovingPeaks(dimensions=2, npeaks=[2, 3, 5], change_severity=1.0)

    landscape.change_peaks()

    assert len(landscape.peaks_function) == 2
    flat_positions = [coord for position in landscape.peaks_position for coord in position]
    assert flat_positions == pytest.approx(
        [80.7428994728303, 9.870377604514095, 37.88377359506277, 78.2789375035795]
    )
    assert landscape.peaks_height == pytest.approx([65.7889272375645, 58.35341778722833])
    assert landscape.peaks_width == pytest.approx([0.09406579248049617, 0.09793301500319732])


def test_change_peaks_is_stable_with_a_fixed_count():
    tools.rng.seed(5)
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


def test_pf1_squared_form_differs_from_euclidean():
    value = tools.MPFuncs.pf1((0.0, 3.0), (0.0, 0.0), 50.0, 0.1)
    squared = 50.0 / (1.0 + 0.1 * 9.0)
    euclidean = 50.0 / (1.0 + 0.1 * 3.0)
    assert value == pytest.approx(squared)
    assert value != pytest.approx(euclidean)


def test_pf1_uses_squared_distance_and_alt1_move_severity():
    value = tools.MPFuncs.pf1((0.0, 3.0), (0.0, 0.0), 50.0, 0.1)
    assert value == pytest.approx(50.0 / (1.0 + 0.1 * 9.0))
    assert tools.MPConfigs.ALT1["move_severity"] == 1.5


def test_pf2_and_pf3_match_closed_forms():
    assert tools.MPFuncs.pf2((3.0, 0.0), (0.0, 0.0), 50.0, 2.0) == pytest.approx(44.0)
    assert tools.MPFuncs.pf3((3.0, 4.0), (0.0, 0.0), 2.0) == pytest.approx(50.0)


def test_multiple_peak_functions_and_random_heights_widths():
    tools.rng.seed(3)
    landscape = MovingPeaks(
        dimensions=2,
        npeaks=2,
        pfunc=[tools.MPFuncs.pf1, tools.MPFuncs.pf2, tools.MPFuncs.pf3],
        uniform_height=0,
        uniform_width=0,
        period=1,
        bfunc=lambda _x: 1.0,
    )

    first = landscape([10.0, 10.0])
    second = landscape([10.0, 10.0], count=False)

    assert landscape.current_error is not None
    assert landscape.offline_error >= 0.0
    assert first[0] >= 1.0
    assert isinstance(second[0], float)
    assert landscape.sorted_maxima
    assert len(landscape.peaks_function) == 2


def test_matching_peak_function_list_is_kept_in_order():
    funcs = [tools.MPFuncs.pf1, tools.MPFuncs.pf2]
    landscape = MovingPeaks(dimensions=1, npeaks=2, pfunc=funcs)

    assert landscape.peaks_function == funcs
    assert landscape.pfunc_pool == tuple(funcs)

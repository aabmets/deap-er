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
from deap_er import gp
from numpy.lib.stride_tricks import sliding_window_view

WINDOW = 129


def spiked_series():
    # One ~3.4e300 sample: its squared deviation overflows, while the
    # window's standard deviation (~3e299) is a finite float.
    series = numpy.random.default_rng(3).normal(100.0, 1.0, 200)
    series[150] = 3.4e300
    return series


def tiny_series():
    # Deviations of ~1e-170 square to below the smallest subnormal.
    return numpy.random.default_rng(4).normal(0.0, 1e-170, 200)


def partner_series():
    return numpy.random.default_rng(5).normal(5.0, 2.0, 200)


def unit_windows(series):
    # Each window divided by its own peak, so the reference moments
    # never leave the float range.
    views = sliding_window_view(series, WINDOW)
    unit = numpy.max(numpy.abs(views), axis=-1)
    return views / unit[:, None], unit


def centered(series):
    views, unit = unit_windows(series)
    return views - views.mean(axis=-1, keepdims=True), unit


def expected_std(series):
    dev, unit = centered(series)
    return numpy.sqrt((dev * dev).mean(axis=-1)) * unit


def expected_pair(name, left, right):
    dev_x, unit_x = centered(left)
    dev_y, unit_y = centered(right)
    cov = (dev_x * dev_y).mean(axis=-1)
    var_x = (dev_x * dev_x).mean(axis=-1)
    var_y = (dev_y * dev_y).mean(axis=-1)
    if name == "rolling_cov":
        return cov * unit_x * unit_y
    if name == "rolling_corr":
        return cov / numpy.sqrt(var_x * var_y)
    return cov / var_y * unit_x / unit_y


@pytest.mark.parametrize("make", [spiked_series, tiny_series])
def test_rolling_std_of_an_extreme_scale_window_is_its_true_value(make):
    series = make()
    actual = gp.rolling_std(series, WINDOW)
    assert numpy.isnan(actual[: WINDOW - 1]).all()
    numpy.testing.assert_allclose(actual[WINDOW - 1 :], expected_std(series), rtol=1e-12)


@pytest.mark.parametrize("make", [spiked_series, tiny_series])
@pytest.mark.parametrize("name", ["rolling_cov", "rolling_corr", "rolling_beta"])
def test_pair_stats_of_an_extreme_scale_window_are_their_true_values(make, name):
    left = make()
    right = partner_series()
    actual = getattr(gp, name)(left, right, WINDOW)
    expected = expected_pair(name, left, right)
    numpy.testing.assert_allclose(actual[WINDOW - 1 :], expected, rtol=1e-12)


@pytest.mark.parametrize("make", [spiked_series, tiny_series])
def test_beta_on_an_extreme_scale_regressor_is_its_true_value(make):
    right = make()
    left = partner_series()
    actual = gp.rolling_beta(left, right, WINDOW)
    expected = expected_pair("rolling_beta", left, right)
    numpy.testing.assert_allclose(actual[WINDOW - 1 :], expected, rtol=1e-12)


def test_ordinary_windows_keep_their_bits():
    # The scaling only touches windows whose squares leave the safe
    # range, so a unit-scale series reduces exactly as before.
    series = partner_series()
    views = sliding_window_view(series, WINDOW)
    dev = views - numpy.add.reduce(views, axis=-1)[:, None] / WINDOW
    residue = numpy.add.reduce(dev, axis=-1) / WINDOW
    squares = numpy.add.reduce(dev * dev, axis=-1) / WINDOW
    expected = numpy.sqrt(numpy.maximum(squares - residue * residue, 0.0))
    numpy.testing.assert_array_equal(gp.rolling_std(series, WINDOW)[WINDOW - 1 :], expected)

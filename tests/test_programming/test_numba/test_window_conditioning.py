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

from .window_parity import PAIR, check_parity, window_kit, window_tree

pytestmark = pytest.mark.skipif(
    not gp.numba_available(), reason="the optional numba extra is not installed"
)


def test_large_magnitude_rolling_std_matches_the_python_oracle():
    # Running sums lose the digits a variance is made of when the level
    # is ~1e8 and the window then holds still.
    series = numpy.concatenate([numpy.linspace(1e8, 1e8 + 80, 40), numpy.full(40, 1e8)])
    pset = window_kit(16, "unary")
    check_parity(window_tree(pset, "rolling_std", "unary"), pset, (series,))


@pytest.mark.parametrize("name", PAIR)
def test_a_large_ramp_then_constant_pair_matches_the_python_oracle(name):
    # After a 1e6 ramp the right series is constant, so its variance is
    # 0 and corr / beta are nan.
    left = numpy.linspace(0.0, 10.0, 64)
    right = numpy.concatenate([numpy.linspace(1e6, 1e6 + 50.0, 32), numpy.full(32, 1e6)])
    pset = window_kit(16, "pair")
    check_parity(window_tree(pset, name, "pair"), pset, (left, right))


def test_a_long_ramp_far_from_its_first_sample_matches_the_python_oracle():
    # Every sample that flows through the window leaves rounding behind,
    # and the window sits far from the first sample.
    pset = window_kit(16, "unary")
    check_parity(window_tree(pset, "rolling_std", "unary"), pset, (numpy.linspace(0.0, 1e5, 5000),))


@pytest.mark.parametrize(("kind", "names"), [("unary", ["rolling_std"]), ("pair", PAIR)])
def test_a_variance_collapse_far_from_the_first_sample_matches_the_oracle(kind, names):
    # The series climbs to 1e8 and then holds still, so its window
    # variance ends up buried far below the magnitude of its samples.
    generator = numpy.random.default_rng(22)
    steady = 1e8 + generator.normal(0.0, 1e-4, 300)
    climb = numpy.concatenate([numpy.linspace(0.0, 1e8, 300), steady])
    plain = numpy.linspace(0.0, 10.0, 600)
    pset = window_kit(16, kind)
    for name in names:
        columns = (climb, plain) if kind == "unary" else (plain, climb)
        check_parity(window_tree(pset, name, kind), pset, columns)


@pytest.mark.parametrize("name", PAIR)
def test_uncorrelated_large_columns_agree_to_an_absolute_floor(name):
    # A covariance near zero is a sum that cancels, so how many digits
    # survive depends on the order the terms are added in. The backends
    # sum a window differently on purpose, so only absolute agreement
    # is a contract here; the relative value is not reproducible.
    generator = numpy.random.default_rng(5)
    left = 1e6 + numpy.cumsum(generator.normal(0.0, 1.0, 20_000))
    right = 1e6 + numpy.cumsum(generator.normal(0.0, 1.0, 20_000))
    pset = window_kit(16, "pair")
    check_parity(window_tree(pset, name, "pair"), pset, (left, right), atol=1e-9)

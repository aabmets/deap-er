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

from .window_parity import check_parity, window_kit, window_span, window_tree

pytestmark = pytest.mark.skipif(
    not gp.numba_available(), reason="the optional numba extra is not installed"
)

ROLLING = ["rolling_sum", "rolling_mean"]


def _squared_tree(pset, name):
    mapping = pset.mapping
    nodes = [mapping[name], mapping["vmul"], mapping["first"], mapping["first"], window_span(pset)]
    return gp.PrimitiveTree(nodes)


@pytest.mark.parametrize("name", ROLLING)
def test_one_huge_sample_does_not_corrupt_later_windows(name):
    # A 1e17 sample wipes the low digits of a running total. Once it has
    # left the window, every later window must again match the oracle.
    series = numpy.random.default_rng(3).normal(size=2000)
    series[100] = 1e17
    pset = window_kit(5, "unary")
    tree = window_tree(pset, name, "unary")
    expected = numpy.asarray(gp.compile_tree(tree, pset)(series, series, series))
    actual = gp.compile_tree(tree, pset, backend="numba")(series, series, series)

    numpy.testing.assert_allclose(actual[200:], expected[200:], rtol=1e-9, atol=1e-12)
    check_parity(tree, pset, (series,))


@pytest.mark.parametrize("name", ROLLING)
def test_a_packed_symbol_at_another_scale_matches_that_symbol_alone(name):
    # A large symbol, NaN padding, then a unit-scale symbol: the padding
    # empties the window, so nothing of the first symbol may leak on.
    generator = numpy.random.default_rng(7)
    large = 1e5 * generator.normal(size=1000)
    small = generator.normal(size=1000)
    packed = numpy.concatenate([large, numpy.full(48, numpy.nan), small])
    pset = window_kit(4, "unary")
    tree = _squared_tree(pset, name)
    run = gp.compile_tree(tree, pset, backend="numba")

    tail = run(packed, packed, packed)[-small.size :]
    numpy.testing.assert_array_equal(tail, run(small, small, small))
    expected = numpy.asarray(gp.compile_tree(tree, pset)(small, small, small))
    numpy.testing.assert_allclose(tail, expected, equal_nan=True, rtol=1e-12, atol=1e-12)


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.parametrize("name", [*ROLLING, "rolling_std"])
def test_windows_after_an_overflowing_window_match_the_python_oracle(name):
    # Two samples near the float maximum overflow the running sums to
    # inf. Once they have left, the sums must come back to finite.
    series = numpy.arange(40.0)
    series[[10, 11]] = 1.5e308
    series[20] = -1e308
    pset = window_kit(3, "unary")
    check_parity(window_tree(pset, name, "unary"), pset, (series,))


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("rolling_std", "unary"),
        ("rolling_corr", "pair"),
        ("rolling_cov", "pair"),
        ("rolling_beta", "pair"),
    ],
)
def test_moment_windows_depend_only_on_the_samples_in_the_window(name, kind):
    # The same window must give the same bits whatever came before it.
    generator = numpy.random.default_rng(7)
    first = numpy.cumsum(generator.normal(size=20_000)) + 100.0
    second = 0.5 * first + numpy.cumsum(generator.normal(size=20_000))
    pset = window_kit(48, kind)
    run = gp.compile_tree(window_tree(pset, name, kind), pset, backend="numba")
    start = 15_000

    full = run(first, second, first)[start + 47 :]
    tail = run(first[start:], second[start:], first[start:])[47:]
    numpy.testing.assert_array_equal(full, tail)

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

from .window_parity import (
    INF,
    NAN,
    PAIR,
    TS_OPS,
    UNARY,
    check_parity,
    window_kit,
    window_span,
    window_tree,
)

pytestmark = pytest.mark.skipif(
    not gp.numba_available(), reason="the optional numba extra is not installed"
)


@pytest.mark.parametrize("name", UNARY)
def test_constructed_unary_trees_match_the_python_oracle(name):
    pset = window_kit(4, "unary")
    check_parity(window_tree(pset, name, "unary"), pset, (numpy.linspace(-1.0, 2.0, 16),))


@pytest.mark.parametrize("name", PAIR)
def test_constructed_pair_trees_match_the_python_oracle(name):
    pset = window_kit(4, "pair")
    left = numpy.linspace(1.0, 4.0, 16)
    check_parity(window_tree(pset, name, "pair"), pset, (left, left[::-1].copy()))


@pytest.mark.parametrize("name", TS_OPS)
def test_constructed_ts_trees_match_the_python_oracle(name):
    pset = window_kit(4, "ts")
    values = numpy.array([1.0, 5.0, 5.0, 1.0, 1.0, 3.0, 0.0, 4.0])
    check_parity(window_tree(pset, name, "ts"), pset, (values,))


@pytest.mark.parametrize("kind,names", [("unary", UNARY), ("pair", PAIR), ("ts", TS_OPS)])
def test_a_nan_that_leaves_the_window_recovers_finite_values(kind, names):
    first = numpy.array([1.0, 2.0, NAN, 4.0, 5.0, 6.0, 7.0])
    second = numpy.array([2.0, 1.0, 3.0, 4.0, 8.0, 6.0, 5.0])
    pset = window_kit(2, kind)
    for name in names:
        check_parity(window_tree(pset, name, kind), pset, (first, second))


@pytest.mark.parametrize("kind,names", [("unary", UNARY), ("pair", PAIR), ("ts", TS_OPS)])
def test_consecutive_nans_match_the_python_oracle(kind, names):
    first = numpy.array([1.0, NAN, NAN, 4.0, 5.0, 6.0])
    second = numpy.array([2.0, 1.0, 3.0, NAN, 5.0, 6.0])
    pset = window_kit(3, kind)
    for name in names:
        check_parity(window_tree(pset, name, kind), pset, (first, second))


def test_a_constant_right_series_is_nan_for_corr_and_beta():
    pset = window_kit(2, "pair")
    left = numpy.arange(8, dtype=numpy.float64)
    right = numpy.ones(8)
    check_parity(window_tree(pset, "rolling_corr", "pair"), pset, (left, right))
    check_parity(window_tree(pset, "rolling_beta", "pair"), pset, (left, right))


@pytest.mark.parametrize("name", PAIR)
def test_pair_ops_poison_when_only_one_side_is_nan(name):
    pset = window_kit(2, "pair")
    left = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])
    right = numpy.array([2.0, 1.0, NAN, 4.0, 5.0])
    check_parity(window_tree(pset, name, "pair"), pset, (left, right))


@pytest.mark.parametrize("kind,names", [("unary", UNARY), ("pair", PAIR), ("ts", TS_OPS)])
def test_a_window_of_one_matches_the_python_oracle(kind, names):
    first = numpy.arange(8, dtype=numpy.float64)
    pset = window_kit(1, kind)
    for name in names:
        check_parity(window_tree(pset, name, kind), pset, (first, first + 1.0))


@pytest.mark.parametrize("window", [6, 40])
@pytest.mark.parametrize("kind,names", [("unary", UNARY), ("pair", PAIR), ("ts", TS_OPS)])
def test_window_equal_or_beyond_the_series_matches_the_oracle(window, kind, names):
    first = numpy.arange(6, dtype=numpy.float64)
    pset = window_kit(window, kind)
    for name in names:
        check_parity(window_tree(pset, name, kind), pset, (first, first[::-1].copy()))


@pytest.mark.parametrize("name", UNARY)
def test_infinities_match_add_reduce_for_unary_rolls(name):
    pset = window_kit(3, "unary")
    values = numpy.array([1.0, INF, 2.0, 3.0, 4.0, -INF, 5.0])
    check_parity(window_tree(pset, name, "unary"), pset, (values,))


@pytest.mark.parametrize("name", PAIR)
def test_infinities_match_the_python_oracle_for_pair_rolls(name):
    pset = window_kit(3, "pair")
    left = numpy.array([1.0, INF, 2.0, 3.0, 0.0, 4.0])
    right = numpy.array([2.0, 1.0, -INF, 3.0, 5.0, 6.0])
    check_parity(window_tree(pset, name, "pair"), pset, (left, right))


@pytest.mark.parametrize("name", ["rolling_min", "rolling_max", "ts_argmax", "ts_argmin"])
def test_infinity_is_a_valid_extreme(name):
    kind = "ts" if name.startswith("ts_") else "unary"
    pset = window_kit(3, kind)
    values = numpy.array([1.0, 2.0, INF, -INF, 0.0])
    check_parity(window_tree(pset, name, kind), pset, (values,))


def test_ts_arg_keeps_the_newest_tie():
    pset = window_kit(3, "ts")
    values = numpy.array([1.0, 5.0, 5.0, 1.0, 1.0])
    columns = (values,)
    check_parity(window_tree(pset, "ts_argmax", "ts"), pset, columns)
    check_parity(window_tree(pset, "ts_argmin", "ts"), pset, columns)


def test_stacked_rolling_mean_recovers_after_warmup():
    pset = window_kit(20, "unary")
    mapping = pset.mapping
    tree = gp.PrimitiveTree(
        [
            mapping["rolling_mean"],
            mapping["rolling_mean"],
            mapping["first"],
            window_span(pset),
            window_span(pset),
        ]
    )
    values = numpy.linspace(-3.0, 5.0, 80)
    values[11] = NAN
    check_parity(tree, pset, (values,))


@pytest.mark.parametrize(
    "kind,names",
    [("unary", UNARY), ("pair", PAIR), ("ts", ["ts_argmax", "ts_argmin"])],
)
def test_a_long_series_and_large_window_stay_within_tolerance(kind, names):
    generator = numpy.random.default_rng(64)
    first = generator.normal(size=256)
    second = generator.normal(size=256)
    first[40] = NAN
    second[90] = NAN
    pset = window_kit(64, kind)
    for name in names:
        check_parity(window_tree(pset, name, kind), pset, (first, second))

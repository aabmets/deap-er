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
"""Golden-value tests pinning the behaviour of seeded operator runs.

The cases live in ``tests/golden/cases.py`` and the recorded results in the
JSON files beside them. Regenerate with::

    uv run python -m tests.golden._generate

A failure here means library behaviour changed. That is sometimes intended,
but it should never happen by accident during a refactor.

Families named ``exact`` are compared with ``==``: they use only comparisons,
integer arithmetic, and the correctly rounded ``sqrt``, so they are
reproducible bit for bit. Families named ``approx`` route through ``exp``,
``log``, ``sin``, ``cos``, fractional ``pow``, or LAPACK, which may differ by
an ulp between platforms, so they are compared with a relative tolerance.
"""

import pytest
from tests.golden.cases import (
    SEEDS,
    benchmarks_case_approx,
    benchmarks_case_exact,
    ea_drivers_case_exact,
    golden_types,
    gp_crossover_case_exact,
    load,
    moving_peaks_case_approx,
    operators_case_approx,
    operators_case_exact,
    selection_case_exact,
    strategies_case_approx,
)

TOLERANCE = 1e-12


@pytest.fixture
def types():
    with golden_types() as created:
        yield created


def _assert_exact(actual, expected):
    assert actual == expected


def _assert_approx(actual, expected):
    assert sorted(actual) == sorted(expected), "case keys changed"
    for key, values in actual.items():
        assert values == pytest.approx(expected[key], rel=TOLERANCE), key


@pytest.mark.parametrize("seed", SEEDS)
def test_selection_matches_golden(types, seed):
    _assert_exact(selection_case_exact(types, seed), load("selection_exact")[str(seed)])


@pytest.mark.parametrize("seed", SEEDS)
def test_gp_crossover_matches_golden(types, seed):
    _assert_exact(gp_crossover_case_exact(types, seed), load("gp_crossover_exact")[str(seed)])


@pytest.mark.parametrize("seed", SEEDS)
def test_ea_drivers_match_golden(types, seed):
    _assert_exact(ea_drivers_case_exact(types, seed), load("ea_drivers_exact")[str(seed)])


@pytest.mark.parametrize("seed", SEEDS)
def test_integer_mutation_matches_golden(types, seed):
    _assert_exact(operators_case_exact(types, seed), load("operators_exact")[str(seed)])


@pytest.mark.parametrize("seed", SEEDS)
def test_zdt_1_and_2_match_golden(types, seed):
    _assert_exact(benchmarks_case_exact(types, seed), load("benchmarks_exact")[str(seed)])


@pytest.mark.parametrize("seed", SEEDS)
def test_bounded_operators_match_golden(types, seed):
    _assert_approx(operators_case_approx(types, seed), load("operators_approx")[str(seed)])


@pytest.mark.parametrize("seed", SEEDS)
def test_zdt_3_4_and_6_match_golden(types, seed):
    _assert_approx(benchmarks_case_approx(types, seed), load("benchmarks_approx")[str(seed)])


@pytest.mark.parametrize("seed", SEEDS)
def test_moving_peaks_matches_golden(types, seed):
    _assert_approx(moving_peaks_case_approx(types, seed), load("moving_peaks_approx")[str(seed)])


@pytest.mark.parametrize("seed", SEEDS)
def test_cma_strategies_match_golden(types, seed):
    _assert_approx(strategies_case_approx(types, seed), load("strategies_approx")[str(seed)])

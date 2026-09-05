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
from deap_er import gp, tools

NAN = numpy.nan
RAMP = numpy.arange(10, dtype=numpy.float64)

WINDOW_OPS = [
    gp.delay,
    gp.diff,
    gp.rolling_sum,
    gp.rolling_mean,
    gp.rolling_std,
    gp.rolling_min,
    gp.rolling_max,
    gp.ema,
]


def test_delay_reads_only_the_past():
    numpy.testing.assert_array_equal(
        gp.delay(RAMP, 2), [NAN, NAN, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    )


def test_diff_subtracts_a_delayed_copy():
    numpy.testing.assert_array_equal(
        gp.diff(RAMP, 3), [NAN, NAN, NAN, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    )


def test_rolling_reductions_cover_the_trailing_window():
    numpy.testing.assert_allclose(
        gp.rolling_sum(RAMP, 3), [NAN, NAN, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0]
    )
    numpy.testing.assert_allclose(
        gp.rolling_mean(RAMP, 3), [NAN, NAN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    )
    numpy.testing.assert_allclose(
        gp.rolling_min(RAMP, 3), [NAN, NAN, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    )
    numpy.testing.assert_allclose(
        gp.rolling_max(RAMP, 3), [NAN, NAN, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    )
    numpy.testing.assert_allclose(gp.rolling_std(RAMP, 3)[2:], numpy.full(8, numpy.sqrt(2.0 / 3.0)))


def test_rolling_standard_deviation_uses_the_window_as_divisor():
    values = numpy.array([1.0, 3.0, 5.0, 11.0])

    result = gp.rolling_std(values, 4)

    numpy.testing.assert_allclose(result[3], numpy.std(values))


@pytest.mark.parametrize(
    ("func", "warmup"),
    [(func, 4 if func in (gp.delay, gp.diff) else 3) for func in WINDOW_OPS],
)
def test_window_ops_write_a_nan_warmup(func, warmup):
    result = func(RAMP, 4)

    assert numpy.all(numpy.isnan(result[:warmup]))
    assert not numpy.any(numpy.isnan(result[warmup:]))


@pytest.mark.parametrize("func", WINDOW_OPS)
def test_window_ops_never_read_the_future(func):
    cutoff = 6
    original = numpy.linspace(1.0, 4.0, 20)
    perturbed = original.copy()
    perturbed[cutoff:] += 1000.0

    before = func(original, 4)
    after = func(perturbed, 4)

    numpy.testing.assert_array_equal(before[:cutoff], after[:cutoff])


@pytest.mark.parametrize("func", WINDOW_OPS)
def test_window_ops_reject_a_window_below_one(func):
    with pytest.raises(ValueError, match="at least 1"):
        func(RAMP, 0)


@pytest.mark.parametrize("func", WINDOW_OPS)
def test_window_ops_return_all_nan_when_the_window_exceeds_the_series(func):
    assert numpy.all(numpy.isnan(func(RAMP, 40)))


@pytest.mark.parametrize("func", WINDOW_OPS)
def test_window_ops_do_not_alias_their_input(func):
    values = RAMP.copy()

    result = func(values, 2)
    result[:] = 0.0

    numpy.testing.assert_array_equal(values, RAMP)


def test_rolling_reductions_propagate_a_nan_through_its_window():
    values = numpy.array([1.0, 2.0, NAN, 4.0, 5.0, 6.0])

    for func in (gp.rolling_sum, gp.rolling_mean, gp.rolling_std, gp.rolling_min, gp.rolling_max):
        result = func(values, 2)
        numpy.testing.assert_array_equal(
            numpy.isnan(result), [True, False, True, True, False, False]
        )


def test_ema_follows_the_causal_recurrence():
    span = 3
    alpha = 2.0 / (span + 1.0)
    expected = [RAMP[0]]
    for sample in RAMP[1:]:
        expected.append(alpha * sample + (1.0 - alpha) * expected[-1])

    result = gp.ema(RAMP, span)

    assert numpy.all(numpy.isnan(result[: span - 1]))
    numpy.testing.assert_allclose(result[span - 1 :], expected[span - 1 :])


def test_ema_returns_all_nan_for_a_series_without_finite_samples():
    assert numpy.all(numpy.isnan(gp.ema(numpy.full(6, NAN), 2)))


def test_window_primitives_are_registered_as_leaf_typed_operators():
    pset = gp.make_column_pset(["value"])
    gp.add_window_primitives(pset)

    assert len(pset.primitives[gp.Array]) == 8
    assert not pset.primitives[gp.Window]
    for primitive in pset.primitives[gp.Array]:
        assert primitive.args == [gp.Array, gp.Window]


def test_add_window_primitives_rejects_a_name_a_column_would_shadow():
    pset = gp.make_column_pset(["ema"])

    with pytest.raises(ValueError, match="shadow"):
        gp.add_window_primitives(pset)


def test_window_ephemeral_samples_inside_its_bounds():
    pset = gp.make_column_pset(["value"])
    gp.add_window_ephemeral(pset, "WINDOW_OPS_BOUNDS", 3, 7)
    tools.rng.seed(5)

    sampler = pset.terminals[gp.Window][0]
    drawn = {sampler().value for _ in range(200)}

    assert drawn <= {3, 4, 5, 6, 7}
    assert len(drawn) > 1


def test_window_ephemeral_can_be_reused_across_primitive_sets():
    first = gp.make_column_pset(["value"])
    second = gp.make_column_pset(["value"])

    gp.add_window_ephemeral(first, "WINDOW_OPS_SHARED", 2, 4)
    gp.add_window_ephemeral(second, "WINDOW_OPS_SHARED", 2, 4)

    assert first.terminals[gp.Window][0] is second.terminals[gp.Window][0]


def test_window_ephemeral_rejects_a_reused_name_with_other_bounds():
    pset = gp.make_column_pset(["value"])
    gp.add_window_ephemeral(pset, "WINDOW_OPS_CLASH", 2, 4)

    other = gp.make_column_pset(["value"])
    with pytest.raises(ValueError, match="already registered"):
        gp.add_window_ephemeral(other, "WINDOW_OPS_CLASH", 5, 9)


@pytest.mark.parametrize(("low", "high"), [(0, 4), (-1, 4), (6, 2)])
def test_window_ephemeral_rejects_invalid_bounds(low, high):
    pset = gp.make_column_pset(["value"])

    with pytest.raises(ValueError):
        gp.add_window_ephemeral(pset, "WINDOW_OPS_INVALID", low, high)

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


def _pset():
    pset = gp.make_column_pset(["first", "second"])
    gp.add_numpy_primitives(pset)
    gp.add_window_primitives(pset)
    gp.add_pair_window_primitives(pset)
    gp.add_ts_primitives(pset)
    return pset


def _tape(expr):
    pset = _pset()
    return gp.lower_tree(gp.PrimitiveTree.from_string(expr, pset), pset)


def _pack(*columns):
    return numpy.ascontiguousarray(numpy.stack(columns, axis=1))


def _grown(prefix_rows=12, n_new=4):
    series = numpy.arange(prefix_rows + n_new, dtype=numpy.float64)
    other = series * 0.5 + 1.0
    prefix = _pack(series[:prefix_rows], other[:prefix_rows])
    grown = _pack(series, other)
    return prefix, grown, n_new


def _cached_mean():
    prefix, grown, n_new = _grown()
    tape = _tape("rolling_mean(first, 3)")
    cached = gp.interpret_tapes([tape], prefix)
    return prefix, grown, n_new, tape, cached


@pytest.mark.parametrize(
    "expr",
    [
        "vadd(first, second)",
        "rolling_mean(first, 3)",
        "delay(first, 4)",
        "diff(first, 2)",
        "rolling_corr(first, second, 4)",
        "ts_argmin(first, 3)",
        "delay(rolling_mean(first, 5), 3)",
        "ema(first, 5)",
    ],
)
def test_suffix_rescore_matches_the_full_matrix_oracle(expr):
    prefix, grown, _n_new = _grown()
    tape = _tape(expr)
    cached = gp.interpret_tapes([tape], prefix)
    actual = gp.suffix_rescore([tape], grown, cached)
    expected = gp.interpret_tapes([tape], grown)
    numpy.testing.assert_allclose(actual, expected, equal_nan=True)
    numpy.testing.assert_allclose(actual[:, : prefix.shape[0]], cached, equal_nan=True)


def test_suffix_rescore_rejects_a_lookback_shorter_than_the_tape_bound():
    _prefix, grown, _n_new, tape, cached = _cached_mean()
    bound = gp.tape_lookback(tape)
    with pytest.raises(ValueError, match="smaller than the tape bound"):
        gp.suffix_rescore([tape], grown, cached, lookback=bound - 1)


def test_too_short_a_suffix_does_not_match_the_oracle():
    _prefix, grown, n_new, tape, _cached = _cached_mean()
    expected = gp.interpret_tapes([tape], grown)
    short = gp.interpret_tapes([tape], grown[-n_new:])
    assert not numpy.allclose(short, expected[:, -n_new:], equal_nan=True)


def test_suffix_rescore_does_not_alias_the_prefix_cache():
    _prefix, grown, _n_new, tape, cached = _cached_mean()
    actual = gp.suffix_rescore([tape], grown, cached)
    cached[:, 0] = 1e9
    assert actual[0, 0] != 1e9


def test_suffix_rescore_accepts_a_generator_and_n_new_zero():
    prefix, _matrix, _n_new = _grown()
    tape = _tape("delay(first, 2)")
    cached = gp.interpret_tapes([tape], prefix)
    actual = gp.suffix_rescore((item for item in (tape,)), prefix, cached, n_new=0)
    numpy.testing.assert_allclose(actual, cached, equal_nan=True)


def test_suffix_rescore_uses_the_max_lookback_of_the_pack():
    prefix, grown, _n_new = _grown()
    tapes = [_tape("vadd(first, second)"), _tape("delay(first, 4)")]
    cached = gp.interpret_tapes(tapes, prefix)
    actual = gp.suffix_rescore(tapes, grown, cached)
    expected = gp.interpret_tapes(tapes, grown)
    numpy.testing.assert_allclose(actual, expected, equal_nan=True)


def test_suffix_rescore_rejects_a_sequence_of_columns():
    _prefix, grown, _n_new, tape, cached = _cached_mean()
    columns = [grown[:, 0]]
    with pytest.raises(ValueError, match="not a sequence of columns"):
        gp.suffix_rescore([tape], columns, cached)


def test_suffix_rescore_rejects_a_one_dimensional_matrix():
    _prefix, grown, _n_new, tape, cached = _cached_mean()
    vector = grown[:, 0]
    with pytest.raises(ValueError, match="got ndim=1"):
        gp.suffix_rescore([tape], vector, cached)


def test_suffix_rescore_rejects_a_negative_lookback():
    _prefix, grown, _n_new, tape, cached = _cached_mean()
    with pytest.raises(ValueError, match="lookback must be at least 0"):
        gp.suffix_rescore([tape], grown, cached, lookback=-1)


def test_suffix_rescore_rejects_n_new_past_the_matrix():
    _prefix, grown, _n_new, tape, cached = _cached_mean()
    with pytest.raises(ValueError, match="exceeds the matrix length"):
        gp.suffix_rescore([tape], grown, cached, n_new=grown.shape[0] + 1)


def test_suffix_rescore_rejects_a_one_dimensional_prefix():
    _prefix, grown, _n_new, tape, cached = _cached_mean()
    row = cached[0]
    with pytest.raises(ValueError, match="prefix must be"):
        gp.suffix_rescore([tape], grown, row)


def test_suffix_rescore_rejects_a_prefix_with_the_wrong_tape_count():
    _prefix, grown, _n_new, tape, cached = _cached_mean()
    doubled = numpy.vstack([cached, cached])
    with pytest.raises(ValueError, match="prefix has 2 tapes"):
        gp.suffix_rescore([tape], grown, doubled)


def test_suffix_rescore_rejects_a_prefix_shorter_than_kept_rows():
    _prefix, grown, _n_new, tape, cached = _cached_mean()
    with pytest.raises(ValueError, match="need at least"):
        gp.suffix_rescore([tape], grown, cached, n_new=2)


def test_suffix_rescore_returns_an_empty_batch():
    _prefix, grown, _n_new = _grown()
    cached = numpy.empty((0, grown.shape[0] - 4))
    actual = gp.suffix_rescore([], grown, cached)
    assert actual.shape == (0, grown.shape[0])

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
from deap_er.private.programming.tape_lookback import opcode_lookback

COLUMNS = ["first", "second"]


def _pset():
    pset = gp.make_column_pset(COLUMNS)
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


def _consumer_tape():
    return gp.Tape(
        opcodes=numpy.array([int(gp.Opcode.COL_LOAD), gp.USER_BASE], dtype=numpy.int32),
        operands=numpy.array([0, -1], dtype=numpy.int32),
        constants=numpy.empty(0, dtype=numpy.float64),
        columns=2,
        depth=1,
        fill=1.0,
    )


def test_opcode_lookback_matches_the_declared_families():
    assert opcode_lookback(int(gp.Opcode.DELAY), 4) == 4
    assert opcode_lookback(int(gp.Opcode.DIFF), 4) == 4
    assert opcode_lookback(int(gp.Opcode.ROLL_MEAN), 5) == 5
    assert opcode_lookback(int(gp.Opcode.TS_RANK), 6) == 6
    assert opcode_lookback(int(gp.Opcode.EMA), 5) == 4
    assert opcode_lookback(int(gp.Opcode.ADD), -1) == 0


def test_tape_lookback_is_zero_for_a_pointwise_program():
    assert gp.tape_lookback(_tape("vadd(first, second)")) == 0


def test_tape_lookback_uses_the_window_arg_delay_steps_and_ema_warmup():
    assert gp.tape_lookback(_tape("rolling_mean(first, 5)")) == 5
    assert gp.tape_lookback(_tape("delay(first, 3)")) == 3
    assert gp.tape_lookback(_tape("ema(first, 5)")) == 4
    assert gp.tape_lookback(_tape("rolling_corr(first, second, 4)")) == 4
    assert gp.tape_lookback(_tape("ts_rank(first, 6)")) == 6


def test_tape_lookback_composes_along_the_tape():
    assert gp.tape_lookback(_tape("delay(rolling_mean(first, 5), 3)")) == 8
    assert gp.tape_lookback(_tape("vadd(rolling_mean(first, 5), delay(second, 3))")) == 5
    assert gp.tape_lookback(_tape("vwhere(vgt(first, second), delay(first, 2), first)")) == 2


def test_tape_lookback_rejects_a_consumer_opcode():
    with pytest.raises(ValueError, match="no lookback certificate"):
        gp.tape_lookback(_consumer_tape())


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
    prefix, grown, _n_new = _grown()
    tape = _tape("rolling_mean(first, 3)")
    cached = gp.interpret_tapes([tape], prefix)
    bound = gp.tape_lookback(tape)
    with pytest.raises(ValueError, match="smaller than the tape bound"):
        gp.suffix_rescore([tape], grown, cached, lookback=bound - 1)


def test_too_short_a_suffix_does_not_match_the_oracle():
    prefix, grown, n_new = _grown()
    tape = _tape("rolling_mean(first, 3)")
    expected = gp.interpret_tapes([tape], grown)
    short = gp.interpret_tapes([tape], grown[-(n_new):])
    assert not numpy.allclose(short, expected[:, -n_new:], equal_nan=True)


def test_suffix_rescore_does_not_alias_the_prefix_cache():
    prefix, grown, _n_new = _grown()
    tape = _tape("rolling_mean(first, 3)")
    cached = gp.interpret_tapes([tape], prefix)
    actual = gp.suffix_rescore([tape], grown, cached)
    cached[:, 0] = 1e9
    assert actual[0, 0] != 1e9


def test_suffix_rescore_accepts_a_generator_and_n_new_zero():
    prefix, _grown, _n_new = _grown()
    tape = _tape("delay(first, 2)")
    cached = gp.interpret_tapes([tape], prefix)
    actual = gp.suffix_rescore((tape for _ in range(1)), prefix, cached, n_new=0)
    numpy.testing.assert_allclose(actual, cached, equal_nan=True)


def test_suffix_rescore_uses_the_max_lookback_of_the_pack():
    prefix, grown, _n_new = _grown()
    tapes = [_tape("vadd(first, second)"), _tape("delay(first, 4)")]
    cached = gp.interpret_tapes(tapes, prefix)
    actual = gp.suffix_rescore(tapes, grown, cached)
    expected = gp.interpret_tapes(tapes, grown)
    numpy.testing.assert_allclose(actual, expected, equal_nan=True)

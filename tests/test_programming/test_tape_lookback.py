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
    tape = _consumer_tape()
    with pytest.raises(ValueError, match="no lookback certificate"):
        gp.tape_lookback(tape)


def test_opcode_lookback_rejects_a_consumer_opcode():
    with pytest.raises(ValueError, match="no lookback certificate"):
        opcode_lookback(gp.USER_BASE, -1)


def test_opcode_lookback_rejects_an_unknown_opcode():
    with pytest.raises(ValueError, match="no lookback certificate"):
        opcode_lookback(-7, 0)


def test_tape_lookback_rejects_an_empty_tape():
    empty = gp.Tape(
        opcodes=numpy.empty(0, dtype=numpy.int32),
        operands=numpy.empty(0, dtype=numpy.int32),
        constants=numpy.empty(0, dtype=numpy.float64),
        columns=2,
        depth=0,
        fill=1.0,
    )
    with pytest.raises(ValueError, match="leaves no result"):
        gp.tape_lookback(empty)


def test_tape_lookback_rejects_an_underflowing_tape():
    underflow = gp.Tape(
        opcodes=numpy.array([int(gp.Opcode.NEG)], dtype=numpy.int32),
        operands=numpy.array([-1], dtype=numpy.int32),
        constants=numpy.empty(0, dtype=numpy.float64),
        columns=2,
        depth=1,
        fill=1.0,
    )
    with pytest.raises(ValueError, match="underflows"):
        gp.tape_lookback(underflow)

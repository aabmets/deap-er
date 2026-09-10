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


def _consumer_tape():
    return gp.Tape(
        opcodes=numpy.array([int(gp.Opcode.COL_LOAD), gp.USER_BASE], dtype=numpy.int32),
        operands=numpy.array([0, -1], dtype=numpy.int32),
        constants=numpy.empty(0, dtype=numpy.float64),
        columns=2,
        depth=1,
        fill=1.0,
    )


def test_bounds_from_matrix_uses_nan_safe_extrema():
    matrix = numpy.array([[1.0, numpy.nan], [3.0, 2.0]])
    bounds = gp.bounds_from_matrix(matrix)
    assert bounds[0, 0] == pytest.approx(1.0)
    assert bounds[0, 1] == pytest.approx(3.0)
    assert bounds[1, 0] == pytest.approx(2.0)
    assert bounds[1, 1] == pytest.approx(2.0)


def test_tape_interval_adds_column_bounds():
    tape = _tape("vadd(first, second)")
    bounds = numpy.array([[0.0, 1.0], [2.0, 4.0]], dtype=numpy.float64)
    lo, hi = gp.tape_interval(tape, bounds)
    assert lo == pytest.approx(2.0)
    assert hi == pytest.approx(5.0)


def test_tape_interval_includes_fill_when_division_crosses_zero():
    tape = _tape("vdiv(first, second)")
    bounds = numpy.array([[-1.0, 1.0], [-2.0, 2.0]], dtype=numpy.float64)
    lo, hi = gp.tape_interval(tape, bounds)
    assert lo <= tape.fill
    assert hi >= tape.fill


def test_tape_interval_keeps_window_output_inside_the_input_hull():
    tape = _tape("rolling_mean(first, 5)")
    bounds = numpy.array([[1.0, 3.0], [0.0, 0.0]], dtype=numpy.float64)
    lo, hi = gp.tape_interval(tape, bounds)
    assert lo >= 1.0
    assert hi <= 3.0


def test_tape_flags_marks_short_matrices_as_all_nan():
    tape = _tape("rolling_mean(first, 8)")
    bounds = numpy.array([[0.0, 1.0], [0.0, 1.0]], dtype=numpy.float64)
    short = gp.tape_flags(tape, bounds, n_rows=6)
    long = gp.tape_flags(tape, bounds, n_rows=16)
    assert short.all_nan
    assert not long.all_nan


def test_tape_flags_marks_a_constant_column_load():
    tape = _tape("first")
    bounds = numpy.array([[2.5, 2.5], [0.0, 1.0]], dtype=numpy.float64)
    flags = gp.tape_flags(tape, bounds, n_rows=8)
    assert flags.constant
    assert flags.skip_score


def test_tape_flags_marks_nan_plus_finite_as_all_nan():
    tape = _tape("vadd(first, second)")
    bounds = numpy.array([[numpy.nan, numpy.nan], [0.0, 1.0]], dtype=numpy.float64)
    flags = gp.tape_flags(tape, bounds, n_rows=8)
    assert flags.all_nan
    assert flags.skip_score


def test_tape_flags_marks_vwhere_that_hides_warmup():
    tape = _tape("vwhere(vgt(rolling_mean(first, 5), second), first, second)")
    bounds = numpy.array([[0.0, 1.0], [0.0, 1.0]], dtype=numpy.float64)
    flags = gp.tape_flags(tape, bounds, n_rows=16)
    assert flags.hides_warmup
    assert flags.skip_score


def test_tape_flags_leaves_pointwise_vwhere_alone():
    tape = _tape("vwhere(vgt(first, second), first, second)")
    bounds = numpy.array([[0.0, 1.0], [0.0, 1.0]], dtype=numpy.float64)
    flags = gp.tape_flags(tape, bounds, n_rows=16)
    assert not flags.hides_warmup


def test_tape_flags_accepts_boolean_terminals_as_masks():
    bounds = numpy.array([[0.0, 1.0], [0.0, 1.0]], dtype=numpy.float64)
    where_false = _tape("vwhere(False, first, second)")
    where_not = _tape("vwhere(vnot(True), first, second)")
    assert not gp.tape_flags(where_false, bounds, n_rows=16).hides_warmup
    assert not gp.tape_flags(where_not, bounds, n_rows=16).hides_warmup


def test_tape_interval_envelope_covers_the_oracle_on_finite_samples():
    tape = _tape("vadd(rolling_mean(first, 3), delay(second, 2))")
    matrix = numpy.column_stack([numpy.linspace(0.0, 1.0, 12), numpy.linspace(2.0, 3.0, 12)])
    bounds = gp.bounds_from_matrix(matrix)
    lo, hi = gp.tape_interval(tape, bounds)
    predicted = gp.interpret_tape(tape, matrix)
    finite = predicted[numpy.isfinite(predicted)]
    assert lo <= finite.min()
    assert hi >= finite.max()


def test_tape_flags_rejects_a_consumer_opcode():
    tape = _consumer_tape()
    bounds = numpy.zeros((2, 2))
    with pytest.raises(ValueError, match="no interval certificate"):
        gp.tape_flags(tape, bounds, n_rows=4)


def test_tape_interval_rejects_an_underflowing_tape():
    underflow = gp.Tape(
        opcodes=numpy.array([int(gp.Opcode.NEG)], dtype=numpy.int32),
        operands=numpy.array([-1], dtype=numpy.int32),
        constants=numpy.empty(0, dtype=numpy.float64),
        columns=2,
        depth=1,
        fill=1.0,
    )
    bounds = numpy.zeros((2, 2))
    with pytest.raises(ValueError, match="underflows"):
        gp.tape_interval(underflow, bounds)

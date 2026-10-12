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
import pickle

import numpy
import pytest
from deap_er import gp, tools

BACKENDS = ("python", "opcode", "numba") if gp.numba_available() else ("python", "opcode")


def _pset(strides, names=("x", "s")):
    return gp.columnar_pset(list(names), window=None, ts=True, pair_windows=True, strides=strides)


def _held(rows, stride, phase=0, seed=3):
    """Return a column that changes every ``stride`` rows, starting mid-step by ``phase``."""
    steps = numpy.random.default_rng(seed).normal(size=rows // stride + 2)
    return numpy.repeat(steps, stride)[phase : phase + rows]


def _run(text, pset, columns, backend):
    tree = gp.PrimitiveTree.from_string(text, pset)
    return numpy.asarray(gp.compile_tree(tree, pset, backend=backend)(*columns))


def _operand(text, pset):
    return int(gp.lower_tree(text, pset).operands[-1])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("phase", [0, 1, 3])
def test_delay_one_reaches_the_previous_step(backend, phase):
    pset = _pset({"s": 4})
    held = _held(40, 4, phase)
    columns = (numpy.arange(40.0), held)
    result = _run("delay(s, 1)", pset, columns, backend)
    numpy.testing.assert_array_equal(result[4:], held[:-4])
    assert numpy.isnan(result[:4]).all()
    # Every defined row holds the value of the step before the row's step.
    step = (numpy.arange(40) + phase) // 4
    for row in range(4, 40):
        assert result[row] == held[numpy.flatnonzero(step == step[row] - 1)[0]]


@pytest.mark.parametrize("backend", BACKENDS)
def test_diff_counts_steps_and_is_zero_nowhere_inside_a_step(backend):
    pset = _pset({"s": 4})
    held = _held(40, 4, phase=2)
    result = _run("diff(s, 2)", pset, (numpy.arange(40.0), held), backend)
    numpy.testing.assert_array_equal(result[8:], held[8:] - held[:-8])
    assert numpy.isnan(result[:8]).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_window_on_an_unstrided_column_still_counts_rows(backend):
    plain = gp.columnar_pset(["x", "s"], window=None)
    strided = _pset({"s": 4})
    columns = (numpy.arange(24.0) ** 2, _held(24, 4))
    for text in ("delay(x, 1)", "diff(x, 3)", "rolling_mean(x, 2)"):
        numpy.testing.assert_array_equal(
            _run(text, strided, columns, backend), _run(text, plain, columns, backend)
        )


def test_windows_are_scaled_by_the_derived_stride():
    pset = gp.columnar_pset(["a", "b", "c", "x"], window=None, strides={"a": 4, "b": 6, "c": 8})
    cases = {
        "delay(a, 3)": 12,
        "delay(vadd(a, b), 1)": 2,  # gcd(4, 6): the sum changes every 2 rows
        "delay(vmul(a, c), 1)": 4,
        "delay(vsub(a, x), 5)": 5,
        "delay(delay(a, 1), 2)": 8,  # delay keeps the stride of its series
        "delay(diff(c, 1), 1)": 8,
        "delay(rolling_mean(a, 2), 3)": 3,  # rolling output has stride 1
        "rolling_mean(c, 2)": 16,
        "ema(b, 2)": 12,
    }
    for text, rows in cases.items():
        assert _operand(text, pset) == rows, text


def test_pair_and_ts_windows_use_the_gcd_of_their_series():
    pset = _pset({"x": 6, "s": 4})
    assert _operand("rolling_corr(x, s, 3)", pset) == 6
    assert _operand("rolling_cov(s, s, 3)", pset) == 12
    assert _operand("ts_rank(x, 2)", pset) == 12
    assert _operand("delay(ts_argmax(s, 2), 2)", pset) == 2


def test_tape_lookback_reports_rows():
    pset = _pset({"s": 4})
    assert gp.tape_lookback(gp.lower_tree("delay(s, 1)", pset)) == 4
    assert gp.tape_lookback(gp.lower_tree("diff(delay(s, 2), 3)", pset)) == 20
    assert gp.tape_lookback(gp.lower_tree("vadd(delay(s, 2), delay(x, 3))", pset)) == 8


@pytest.mark.parametrize("backend", BACKENDS)
def test_mixed_stride_arithmetic_tree(backend):
    pset = gp.columnar_pset(["a", "b"], window=None, strides={"a": 4, "b": 6})
    a, b = _held(48, 4, seed=5), _held(48, 6, seed=7)
    result = _run("diff(vadd(a, b), 1)", pset, (a, b), backend)
    total = a + b
    numpy.testing.assert_allclose(result[2:], total[2:] - total[:-2])


@pytest.mark.parametrize("backend", ["opcode", "numba"] if gp.numba_available() else ["opcode"])
def test_dropping_lookback_rows_never_reads_another_block(backend):
    pset = _pset({"s": 4})
    text = "vadd(diff(s, 2), delay(x, 1))"
    lookback = gp.tape_lookback(gp.lower_tree(text, pset))
    blocks = [
        (_held(30, 4, phase, seed), numpy.arange(30.0) * (phase + 1))
        for phase, seed in ((0, 1), (3, 2))
    ]
    pad = numpy.full((lookback, 2), numpy.nan)
    parts = []
    for held, fast in blocks:
        parts += [numpy.column_stack([fast, held]), pad]
    matrix = numpy.vstack(parts[:-1])
    packed = gp.interpret_tapes([gp.lower_tree(text, pset)], matrix, backend=backend)[0]
    start = 0
    for held, fast in blocks:
        alone = _run(text, pset, (fast, held), "opcode")
        numpy.testing.assert_array_equal(packed[start + lookback : start + 30], alone[lookback:])
        start += 30 + lookback


def test_text_form_round_trips_and_keeps_steps():
    pset = gp.columnar_pset(["x", "s"], window=(1, 8), strides={"s": 4})
    text = "vadd(delay(s, 3), rolling_mean(x, 2))"
    tree = gp.PrimitiveTree.from_string(text, pset)
    assert str(tree) == text
    assert str(gp.PrimitiveTree.from_string(str(tree), pset)) == text
    loaded = pickle.loads(pickle.dumps(tree))
    assert str(loaded) == text
    numpy.testing.assert_array_equal(
        gp.lower_tree(loaded, pset).operands, gp.lower_tree(tree, pset).operands
    )
    assert pickle.loads(pickle.dumps(pset)).column_strides == (1, 4)


def test_the_window_ephemeral_name_and_draws_are_unchanged():
    plain = gp.columnar_pset(["x", "s"], window=(2, 9))
    strided = gp.columnar_pset(["x", "s"], window=(2, 9), strides={"s": 4})
    for pset in (plain, strided):
        assert [term.__name__ for term in pset.terminals[gp.Window]] == ["window_2_9"]
    draws = []
    for pset in (plain, strided):
        tools.rng.seed(5)
        draws.append([str(gp.PrimitiveTree(gp.gen_grow(pset, 2, 4))) for _ in range(20)])
    assert draws[0] == draws[1]


@pytest.mark.parametrize(
    ("strides", "message"),
    [
        ({"y": 2}, "names no column"),
        ({"s": 0}, "positive integer"),
        ({"s": 2.5}, "positive integer"),
        ({"s": True}, "positive integer"),
    ],
)
def test_bad_strides_are_rejected(strides, message):
    with pytest.raises(ValueError, match=message):
        gp.make_column_pset(["x", "s"], strides=strides)


def test_scaled_window_past_the_operand_range_is_rejected():
    pset = _pset({"s": 2**20})
    with pytest.raises(gp.ProgramError, match="no larger than"):
        gp.lower_tree("delay(s, 4096)", pset)

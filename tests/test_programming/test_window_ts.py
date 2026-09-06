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
TS_OPS = [gp.ts_rank, gp.ts_argmax, gp.ts_argmin]
COLUMNS = ["first", "second", "third"]


def _ts_kit(window_name):
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    gp.add_ts_primitives(pset)
    gp.add_window_ephemeral(pset, window_name, 2, 5)
    return pset


def _samples():
    generator = numpy.random.default_rng(17)
    columns = [
        generator.normal(size=24),
        generator.normal(size=24) * 2.0,
        numpy.abs(generator.normal(size=24)),
    ]
    columns[1][3] = 0.0
    columns[2][7] = numpy.nan
    return tuple(columns)


def _as_column(value, size):
    return numpy.broadcast_to(numpy.asarray(value, dtype=numpy.float64), (size,))


def test_ts_rank_scales_low_high_and_ties_to_the_unit_interval():
    values = numpy.array([1.0, 3.0, 2.0, 2.0, 0.0])

    result = gp.ts_rank(values, 3)

    assert numpy.all(numpy.isnan(result[:2]))
    numpy.testing.assert_allclose(result[2:], [0.5, 0.25, 0.0])
    numpy.testing.assert_allclose(gp.ts_rank(numpy.array([1.0, 2.0, 3.0]), 3)[2], 1.0)
    numpy.testing.assert_allclose(gp.ts_rank(numpy.array([2.0, 2.0, 2.0]), 3)[2], 0.5)


def test_ts_arg_reports_bars_ago_and_keeps_the_newest_tie():
    values = numpy.array([1.0, 5.0, 5.0, 1.0, 1.0])

    numpy.testing.assert_allclose(gp.ts_argmax(values, 3)[2:], [0.0, 1.0, 2.0])
    numpy.testing.assert_allclose(gp.ts_argmin(values, 3)[2:], [2.0, 0.0, 0.0])


@pytest.mark.parametrize("func", TS_OPS)
def test_ts_ops_write_a_nan_warmup(func):
    result = func(RAMP, 4)

    assert numpy.all(numpy.isnan(result[:3]))
    assert not numpy.any(numpy.isnan(result[3:]))


def test_ts_rank_of_a_one_sample_window_is_nan():
    assert numpy.all(numpy.isnan(gp.ts_rank(RAMP, 1)))


def test_ts_arg_of_a_one_sample_window_is_the_current_bar():
    numpy.testing.assert_allclose(gp.ts_argmax(RAMP, 1), numpy.zeros(10))
    numpy.testing.assert_allclose(gp.ts_argmin(RAMP, 1), numpy.zeros(10))


@pytest.mark.parametrize("func", TS_OPS)
def test_ts_ops_never_read_the_future(func):
    original = numpy.linspace(1.0, 4.0, 20)
    perturbed = original.copy()
    perturbed[6:] += 1000.0

    numpy.testing.assert_array_equal(func(original, 4)[:6], func(perturbed, 4)[:6])


@pytest.mark.parametrize("func", TS_OPS)
def test_ts_ops_reject_a_window_below_one(func):
    with pytest.raises(ValueError, match="at least 1"):
        func(RAMP, 0)


@pytest.mark.parametrize("func", TS_OPS)
def test_ts_ops_return_all_nan_when_the_window_exceeds_the_series(func):
    assert numpy.all(numpy.isnan(func(RAMP, 40)))


@pytest.mark.parametrize("func", TS_OPS)
def test_ts_ops_do_not_alias_their_input(func):
    values = RAMP.copy()

    result = func(values, 2)
    result[:] = 0.0

    numpy.testing.assert_array_equal(values, RAMP)


def test_ts_ops_propagate_a_nan_through_its_window():
    values = numpy.array([1.0, 2.0, NAN, 4.0, 5.0, 6.0])

    for func in TS_OPS:
        numpy.testing.assert_array_equal(
            numpy.isnan(func(values, 2)), [True, False, True, True, False, False]
        )


def test_ts_ops_treat_infinity_as_a_valid_extreme():
    values = numpy.array([1.0, 2.0, numpy.inf])

    numpy.testing.assert_allclose(gp.ts_rank(values, 3)[2], 1.0)
    numpy.testing.assert_allclose(gp.ts_argmax(values, 3)[2], 0.0)
    numpy.testing.assert_allclose(gp.ts_argmin(numpy.array([-numpy.inf, 1.0, 2.0]), 3)[2], 2.0)


def test_ts_primitives_are_registered_as_leaf_typed_operators():
    pset = gp.make_column_pset(["value"])
    gp.add_ts_primitives(pset)

    assert len(pset.primitives[gp.Array]) == 3
    for primitive in pset.primitives[gp.Array]:
        assert primitive.args == [gp.Array, gp.Window]


def test_add_ts_primitives_rejects_a_name_a_column_would_shadow():
    pset = gp.make_column_pset(["ts_rank"])

    with pytest.raises(ValueError, match="shadow"):
        gp.add_ts_primitives(pset)


def test_lowering_folds_a_ts_window_into_an_immediate_operand():
    pset = _ts_kit("TS_WINDOW_IMMEDIATE")
    mapping = pset.mapping
    window = pset.terminals[gp.Window][0]
    tree = gp.PrimitiveTree([mapping["ts_rank"], mapping["first"], window()])

    tape = gp.lower_tree(tree, pset)

    assert list(tape.opcodes) == [gp.Opcode.COL_LOAD, gp.Opcode.TS_RANK]
    assert tape.operands[1] == tree[2].value
    assert tape.depth == 1


@pytest.mark.parametrize("name", ["ts_rank", "ts_argmax", "ts_argmin"])
def test_constructed_ts_trees_match_across_backends(name):
    pset = _ts_kit("TS_WINDOW_CONSTRUCTED")
    mapping = pset.mapping
    window = pset.terminals[gp.Window][0]
    tree = gp.PrimitiveTree([mapping[name], mapping["first"], window()])
    columns = _samples()
    expected = gp.compile_tree(tree, pset)(*columns)
    opcode = gp.compile_tree(tree, pset, backend="opcode")(*columns)
    numpy.testing.assert_allclose(opcode, expected, equal_nan=True)
    if not gp.numba_available():
        return
    numba = gp.compile_tree(tree, pset, backend="numba")(*columns)
    numpy.testing.assert_allclose(numba, expected, equal_nan=True, rtol=1e-9, atol=1e-12)


def test_the_opcode_backend_matches_the_default_backend_on_ts_windows():
    pset = _ts_kit("TS_WINDOW_OPCODE")
    columns = _samples()
    tools.rng.seed(23)

    for _ in range(80):
        tree = gp.PrimitiveTree(gp.gen_half_and_half(pset, 2, 4))
        expected = gp.compile_tree(tree, pset)(*columns)
        actual = gp.compile_tree(tree, pset, backend="opcode")(*columns)
        numpy.testing.assert_allclose(
            _as_column(actual, 24), _as_column(expected, 24), equal_nan=True
        )


@pytest.mark.skipif(not gp.numba_available(), reason="the optional numba extra is not installed")
def test_the_numba_backend_matches_the_default_backend_on_ts_windows():
    pset = _ts_kit("TS_WINDOW_NUMBA")
    columns = _samples()
    tools.rng.seed(29)

    for _ in range(80):
        tree = gp.PrimitiveTree(gp.gen_half_and_half(pset, 2, 4))
        expected = _as_column(gp.compile_tree(tree, pset)(*columns), 24)
        actual = gp.compile_tree(tree, pset, backend="numba")(*columns)
        numpy.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-9, atol=1e-12)

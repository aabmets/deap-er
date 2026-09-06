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
OTHER = numpy.array([2.0, 0.0, 4.0, 1.0, 5.0, 3.0, 7.0, 8.0, 6.0, 9.0])
PAIR_OPS = [gp.rolling_corr, gp.rolling_cov, gp.rolling_beta]
COLUMNS = ["first", "second", "third"]


def _pair_kit(window_name):
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    gp.add_pair_window_primitives(pset)
    gp.add_window_ephemeral(pset, window_name, 1, 5)
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


def test_rolling_pair_matches_population_moments():
    left = numpy.array([1.0, 3.0, 5.0, 11.0])
    right = numpy.array([2.0, 1.0, 4.0, 8.0])
    cov = numpy.cov(left, right, ddof=0)[0, 1]
    var_y = numpy.var(right, ddof=0)

    numpy.testing.assert_allclose(gp.rolling_cov(left, right, 4)[3], cov)
    numpy.testing.assert_allclose(
        gp.rolling_corr(left, right, 4)[3], numpy.corrcoef(left, right)[0, 1]
    )
    numpy.testing.assert_allclose(gp.rolling_beta(left, right, 4)[3], cov / var_y)


def test_self_pair_windows_recover_variance_and_unit_slope():
    cov = gp.rolling_cov(RAMP, RAMP, 3)
    std = gp.rolling_std(RAMP, 3)
    corr = gp.rolling_corr(RAMP, RAMP, 3)
    beta = gp.rolling_beta(RAMP, RAMP, 3)

    numpy.testing.assert_allclose(cov[2:], std[2:] ** 2)
    numpy.testing.assert_allclose(corr[2:], numpy.ones(8))
    numpy.testing.assert_allclose(beta[2:], numpy.ones(8))


def test_rolling_beta_is_not_symmetric():
    assert not numpy.allclose(
        gp.rolling_beta(RAMP, OTHER, 4)[3:], gp.rolling_beta(OTHER, RAMP, 4)[3:]
    )


@pytest.mark.parametrize("func", PAIR_OPS)
def test_pair_ops_write_a_nan_warmup(func):
    result = func(RAMP, OTHER, 4)

    assert numpy.all(numpy.isnan(result[:3]))
    assert not numpy.any(numpy.isnan(result[3:]))


@pytest.mark.parametrize("func", PAIR_OPS)
@pytest.mark.parametrize("side", [0, 1])
def test_pair_ops_never_read_the_future(func, side):
    original_left = numpy.linspace(1.0, 4.0, 20)
    original_right = numpy.linspace(2.0, 6.0, 20)
    left, right = original_left.copy(), original_right.copy()
    (left if side == 0 else right)[6:] += 1000.0

    numpy.testing.assert_array_equal(
        func(original_left, original_right, 4)[:6], func(left, right, 4)[:6]
    )


@pytest.mark.parametrize("func", PAIR_OPS)
def test_pair_ops_reject_a_window_below_one(func):
    with pytest.raises(ValueError, match="at least 1"):
        func(RAMP, OTHER, 0)


@pytest.mark.parametrize("func", PAIR_OPS)
def test_pair_ops_return_all_nan_when_the_window_exceeds_the_series(func):
    assert numpy.all(numpy.isnan(func(RAMP, OTHER, 40)))


@pytest.mark.parametrize("func", PAIR_OPS)
def test_pair_ops_do_not_alias_their_input(func):
    left, right = RAMP.copy(), OTHER.copy()

    result = func(left, right, 2)
    result[:] = 0.0

    numpy.testing.assert_array_equal(left, RAMP)
    numpy.testing.assert_array_equal(right, OTHER)


@pytest.mark.parametrize("func", PAIR_OPS)
@pytest.mark.parametrize("right", [OTHER[:4], OTHER.reshape(10, 1), OTHER.reshape(2, 5)])
def test_pair_ops_reject_unequal_lengths(func, right):
    with pytest.raises(ValueError, match="same length"):
        func(RAMP, right, 2)


def test_rolling_pair_propagates_a_nan_from_either_series():
    left = numpy.array([1.0, 2.0, NAN, 4.0, 5.0, 6.0])
    right = numpy.array([2.0, 1.0, 3.0, 4.0, NAN, 6.0])

    for func in PAIR_OPS:
        numpy.testing.assert_array_equal(
            numpy.isnan(func(left, right, 2)), [True, False, True, True, True, True]
        )


def test_zero_variance_window_is_nan_for_corr_and_beta():
    constant = numpy.ones(6)
    ramp = numpy.arange(6, dtype=numpy.float64)

    numpy.testing.assert_array_equal(numpy.isnan(gp.rolling_corr(ramp, constant, 3)[2:]), True)
    numpy.testing.assert_array_equal(numpy.isnan(gp.rolling_beta(ramp, constant, 3)[2:]), True)
    numpy.testing.assert_allclose(gp.rolling_cov(ramp, constant, 3)[2:], 0.0)


def test_window_of_one_is_zero_cov_and_undefined_corr_beta():
    numpy.testing.assert_allclose(gp.rolling_cov(RAMP, OTHER, 1), numpy.zeros(10))
    assert numpy.all(numpy.isnan(gp.rolling_corr(RAMP, OTHER, 1)))
    assert numpy.all(numpy.isnan(gp.rolling_beta(RAMP, OTHER, 1)))


def test_pair_primitives_are_registered_as_two_array_operators():
    pset = gp.make_column_pset(["first", "second"])
    gp.add_pair_window_primitives(pset)

    assert len(pset.primitives[gp.Array]) == 3
    for primitive in pset.primitives[gp.Array]:
        assert primitive.args == [gp.Array, gp.Array, gp.Window]


def test_add_pair_window_primitives_rejects_a_name_a_column_would_shadow():
    pset = gp.make_column_pset(["rolling_corr"])

    with pytest.raises(ValueError, match="shadow"):
        gp.add_pair_window_primitives(pset)


def test_lowering_folds_a_pair_window_into_an_immediate_operand():
    pset = _pair_kit("PAIR_WINDOW_IMMEDIATE")
    mapping = pset.mapping
    window = pset.terminals[gp.Window][0]
    tree = gp.PrimitiveTree(
        [mapping["rolling_corr"], mapping["first"], mapping["second"], window()]
    )

    tape = gp.lower_tree(tree, pset)

    assert list(tape.opcodes) == [gp.Opcode.COL_LOAD, gp.Opcode.COL_LOAD, gp.Opcode.ROLL_CORR]
    assert tape.operands[2] == tree[3].value
    assert tape.depth == 2


@pytest.mark.parametrize("name", ["rolling_corr", "rolling_cov", "rolling_beta"])
def test_constructed_pair_trees_match_across_backends(name):
    pset = _pair_kit("PAIR_WINDOW_CONSTRUCTED")
    mapping = pset.mapping
    window = pset.terminals[gp.Window][0]
    tree = gp.PrimitiveTree([mapping[name], mapping["first"], mapping["second"], window()])
    columns = _samples()
    expected = gp.compile_tree(tree, pset)(*columns)
    opcode = gp.compile_tree(tree, pset, backend="opcode")(*columns)
    numpy.testing.assert_allclose(opcode, expected, equal_nan=True)
    if not gp.numba_available():
        return
    numba = gp.compile_tree(tree, pset, backend="numba")(*columns)
    numpy.testing.assert_allclose(numba, expected, equal_nan=True, rtol=1e-9, atol=1e-12)


def test_the_opcode_backend_matches_the_default_backend_on_pair_windows():
    pset = _pair_kit("PAIR_WINDOW_OPCODE")
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
def test_the_numba_backend_matches_the_default_backend_on_pair_windows():
    pset = _pair_kit("PAIR_WINDOW_NUMBA")
    columns = _samples()
    tools.rng.seed(29)

    for _ in range(80):
        tree = gp.PrimitiveTree(gp.gen_half_and_half(pset, 2, 4))
        expected = _as_column(gp.compile_tree(tree, pset)(*columns), 24)
        actual = gp.compile_tree(tree, pset, backend="numba")(*columns)
        numpy.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-9, atol=1e-12)

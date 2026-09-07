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
from deap_er.private.programming.numpy.numpy_ops import infer_fill

NAN = numpy.nan


def test_protected_division_fills_only_fabricated_non_finite_values():
    left = numpy.array([1.0, 0.0, NAN, 4.0])
    right = numpy.array([0.0, 0.0, 1.0, 2.0])

    result = gp.vdiv(left, right)

    numpy.testing.assert_array_equal(numpy.isnan(result), [False, False, True, False])
    numpy.testing.assert_allclose(result[[0, 1, 3]], [1.0, 1.0, 2.0])


def test_protected_division_preserves_a_non_finite_divisor():
    result = gp.vdiv(numpy.array([1.0, 1.0]), numpy.array([NAN, numpy.inf]))

    assert numpy.isnan(result[0])
    assert result[1] == 0.0


def test_protected_operations_accept_scalars():
    assert gp.vdiv(1.0, 0.0) == 1.0
    assert gp.vdiv(1.0, 4.0) == 0.25
    assert numpy.isnan(gp.vdiv(NAN, 0.0))
    assert gp.vlog(0.0) == 1.0
    assert gp.vsqrt(-1.0) == 1.0
    assert numpy.isnan(gp.vsqrt(NAN))


def test_protected_operations_take_a_custom_fill():
    numpy.testing.assert_allclose(gp.vdiv(numpy.array([1.0]), numpy.array([0.0]), fill=0.0), [0.0])
    numpy.testing.assert_allclose(gp.vlog(numpy.array([0.0]), fill=-5.0), [-5.0])


def test_protected_logarithm_and_root_guard_their_domain():
    values = numpy.array([1.0, 0.0, -4.0, NAN])

    logs = gp.vlog(values)
    roots = gp.vsqrt(values)

    numpy.testing.assert_allclose(logs[:3], [0.0, 1.0, 1.0])
    numpy.testing.assert_allclose(roots[:3], [1.0, 0.0, 1.0])
    assert numpy.isnan(logs[3])
    assert numpy.isnan(roots[3])


def test_comparisons_return_boolean_masks():
    left = numpy.array([1.0, 2.0, NAN])
    right = numpy.array([2.0, 2.0, 1.0])

    for func, expected in [
        (gp.vgt, [False, False, False]),
        (gp.vlt, [True, False, False]),
        (gp.vge, [False, True, False]),
        (gp.vle, [True, True, False]),
        (gp.veq, [False, True, False]),
    ]:
        mask = func(left, right)
        assert mask.dtype == numpy.bool_
        numpy.testing.assert_array_equal(mask, expected)


def test_mask_logic_and_selection():
    left = numpy.array([True, True, False, False])
    right = numpy.array([True, False, True, False])

    numpy.testing.assert_array_equal(gp.vand(left, right), [True, False, False, False])
    numpy.testing.assert_array_equal(gp.vor(left, right), [True, True, True, False])
    numpy.testing.assert_array_equal(gp.vnot(left), [False, False, True, True])
    numpy.testing.assert_allclose(
        gp.vwhere(left, numpy.arange(4.0), numpy.full(4, -1.0)), [0.0, 1.0, -1.0, -1.0]
    )


def test_add_numpy_primitives_rejects_a_name_a_column_would_shadow():
    pset = gp.make_column_pset(["vmul"])

    with pytest.raises(ValueError, match="shadow"):
        gp.add_numpy_primitives(pset)


def test_infer_fill_reads_back_the_registered_fill():
    default = gp.make_column_pset(["value"])
    custom = gp.make_column_pset(["value"])
    gp.add_numpy_primitives(default)
    gp.add_numpy_primitives(custom, fill=-2.5)

    assert infer_fill(default) == 1.0
    assert infer_fill(custom) == -2.5
    assert infer_fill(gp.make_column_pset(["value"])) == 1.0


def test_mask_terminals_let_generation_close_a_condition_branch():
    pset = gp.make_column_pset(["value"])
    gp.add_numpy_primitives(pset)

    assert len(pset.terminals[gp.Mask]) == 2
    assert {terminal.value for terminal in pset.terminals[gp.Mask]} == {True, False}


def test_generation_over_the_full_kit_never_runs_out_of_terminals():
    pset = gp.make_column_pset(["first", "second"])
    gp.add_numpy_primitives(pset)
    gp.add_window_primitives(pset)
    gp.add_window_ephemeral(pset, "NUMPY_OPS_GEN_WINDOW", 2, 5)
    columns = (numpy.linspace(-2.0, 2.0, 24), numpy.linspace(1.0, 3.0, 24))

    for _ in range(200):
        tree = gp.PrimitiveTree(gp.gen_half_and_half(pset, 1, 4))
        result = gp.compile_tree(tree, pset)(*columns)
        assert numpy.ndim(result) <= 1

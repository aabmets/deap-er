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
from deap_er.gp import numba_ops

pytestmark = pytest.mark.skipif(
    not gp.numba_available(), reason="the optional numba extra is not installed"
)

COLUMNS = ["first", "second", "third"]
TRIPLE = gp.USER_BASE + 61
SCRATCH_USER = gp.USER_BASE + 63


def _kit(window_name):
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    gp.add_window_primitives(pset)
    gp.add_window_ephemeral(pset, window_name, 1, 5)
    return pset


def _samples():
    generator = numpy.random.default_rng(31)
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


def _triple(value):
    return 3.0 * numpy.asarray(value, dtype=numpy.float64)


def _dispatch():
    import numba

    @numba.njit(cache=False, nogil=True, error_model="numpy")
    def dispatch(op, sp, stack, columns, constants, scratch):
        if op == TRIPLE:
            for index in range(columns.shape[0]):
                stack[sp - 1, index] = 3.0 * stack[sp - 1, index]
            return sp
        return -1

    return dispatch


def test_the_numba_backend_matches_the_default_backend():
    pset = _kit("NUMBA_OPS_PARITY")
    columns = _samples()
    tools.seed(29)

    for _ in range(250):
        tree = gp.PrimitiveTree(gp.gen_half_and_half(pset, 2, 4))
        expected = _as_column(gp.compile_tree(tree, pset)(*columns), 24)
        actual = gp.compile_tree(tree, pset, backend="numba")(*columns)
        numpy.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-9, atol=1e-12)


def test_the_numba_backend_accepts_a_prepacked_matrix():
    pset = _kit("NUMBA_OPS_MATRIX")
    columns = _samples()
    mapping = pset.mapping
    tree = gp.PrimitiveTree([mapping["vmul"], mapping["first"], mapping["second"]])

    func = gp.compile_tree(tree, pset, backend="numba")

    numpy.testing.assert_allclose(func(numpy.stack(columns, axis=1)), func(*columns))


def test_the_numba_backend_returns_an_independent_array():
    pset = _kit("NUMBA_OPS_INDEPENDENT")
    columns = _samples()
    func = gp.compile_tree(
        gp.PrimitiveTree([pset.mapping["vabs"], pset.mapping["first"]]), pset, backend="numba"
    )

    first = func(*columns)
    first[:] = 0.0
    second = func(*columns)

    assert not numpy.allclose(first, second)


def test_the_numba_backend_rejects_a_wrong_column_count():
    pset = _kit("NUMBA_OPS_ARITY")
    func = gp.compile_tree(gp.PrimitiveTree([pset.mapping["first"]]), pset, backend="numba")

    with pytest.raises(ValueError, match="expects 3 columns"):
        func(numpy.zeros(4), numpy.zeros(4))


def test_a_builtin_tree_needs_no_dispatch_kernel():
    pset = _kit("NUMBA_OPS_BUILTIN")
    columns = _samples()
    mapping = pset.mapping
    tree = gp.PrimitiveTree([mapping["vadd"], mapping["first"], mapping["second"]])

    result = gp.compile_tree(tree, pset, backend="numba")(*columns)

    numpy.testing.assert_allclose(result, columns[0] + columns[1])


def test_a_consumer_kernel_runs_through_the_dispatcher():
    pset = _kit("NUMBA_OPS_DISPATCH")
    pset.add_primitive(_triple, [gp.Array], gp.Array, "numba_ops_triple")
    gp.bind_numba_opcode("numba_ops_triple", TRIPLE)
    mapping = pset.mapping
    tree = gp.PrimitiveTree(
        [mapping["vadd"], mapping["numba_ops_triple"], mapping["first"], mapping["second"]]
    )
    columns = _samples()

    expected = gp.compile_tree(tree, pset)(*columns)
    actual = gp.compile_tree(tree, pset, backend="numba", dispatch=_dispatch())(*columns)

    numpy.testing.assert_allclose(actual, expected, equal_nan=True)


def test_a_consumer_kernel_without_a_dispatcher_is_rejected():
    pset = _kit("NUMBA_OPS_MISSING")
    pset.add_primitive(_triple, [gp.Array], gp.Array, "numba_ops_orphan")
    gp.bind_numba_opcode("numba_ops_orphan", gp.USER_BASE + 62)
    tree = gp.PrimitiveTree([pset.mapping["numba_ops_orphan"], pset.mapping["first"]])

    with pytest.raises(ValueError, match="no dispatch kernel"):
        gp.compile_tree(tree, pset, backend="numba")


def test_the_numba_backend_rejects_a_set_without_arguments():
    # The interpreter sizes every stack row by the column length, so a
    # constant-only set gives it nothing to size the result with.
    pset = gp.PrimitiveSetTyped("MAIN", [], gp.Array)
    gp.add_numpy_primitives(pset)
    pset.add_terminal(2.0, gp.Array, "two")
    tree = gp.PrimitiveTree([pset.mapping["vadd"], pset.mapping["two"], pset.mapping["two"]])

    assert gp.compile_tree(tree, pset, backend="opcode") == 4.0
    with pytest.raises(ValueError, match="cannot size a result"):
        gp.compile_tree(tree, pset, backend="numba")


def test_compiled_tapes_share_one_workspace():
    # A per-callable workspace would let the compile cache pin one
    # large buffer for every program it remembers.
    pset = _kit("NUMBA_OPS_WORKSPACE")
    columns = _samples()
    mapping = pset.mapping
    shallow = gp.PrimitiveTree([mapping["vneg"], mapping["first"]])
    deep = gp.PrimitiveTree(
        [mapping["vadd"], mapping["vmul"], mapping["first"], mapping["second"], mapping["third"]]
    )

    gp.compile_tree(shallow, pset, backend="numba")(*columns)
    held = numba_ops._workspace["stack"]
    gp.compile_tree(deep, pset, backend="numba")(*columns)

    assert numba_ops._workspace["stack"].shape[1] == 24
    assert held.base is None


def test_the_workspace_reserves_the_row_promised_to_a_kernel():
    stack, scratch = numba_ops._reserve(3, 16)

    assert stack.shape == (4, 16)
    assert scratch.shape == (16,)
    assert stack.flags["C_CONTIGUOUS"]


def test_a_kernel_may_use_the_free_stack_row():
    pset = _kit("NUMBA_OPS_FREE_ROW")
    pset.add_primitive(_triple, [gp.Array], gp.Array, "numba_ops_scratch_user")
    gp.bind_numba_opcode("numba_ops_scratch_user", SCRATCH_USER)
    mapping = pset.mapping
    tree = gp.PrimitiveTree([mapping["numba_ops_scratch_user"], mapping["first"]])
    columns = _samples()

    import numba

    @numba.njit(cache=False, nogil=True, error_model="numpy")
    def dispatch(op, sp, stack, columns, constants, scratch):
        if op == SCRATCH_USER:
            for index in range(columns.shape[0]):
                stack[sp, index] = 3.0 * stack[sp - 1, index]
            for index in range(columns.shape[0]):
                stack[sp - 1, index] = stack[sp, index]
            return sp
        return -1

    result = gp.compile_tree(tree, pset, backend="numba", dispatch=dispatch)(*columns)

    numpy.testing.assert_allclose(result, 3.0 * columns[0], equal_nan=True)


def test_the_dispatch_signature_is_documented():
    assert "stack" in gp.USER_DISPATCH_SIGNATURE
    assert "scratch" in gp.USER_DISPATCH_SIGNATURE


def test_bind_tape_is_usable_on_its_own():
    pset = _kit("NUMBA_OPS_DIRECT")
    columns = _samples()
    tape = gp.lower_tree(gp.PrimitiveTree([pset.mapping["vneg"], pset.mapping["third"]]), pset)

    numpy.testing.assert_allclose(gp.bind_tape(tape)(*columns), -columns[2], equal_nan=True)

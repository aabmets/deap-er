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
from deap_er.private.programming.numba import numba_ops

pytestmark = pytest.mark.skipif(
    not gp.numba_available(), reason="the optional numba extra is not installed"
)

COLUMNS = ["first", "second", "third"]
TRIPLE = gp.USER_BASE + 81


def _kit(window_name):
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    gp.add_window_primitives(pset)
    gp.add_pair_window_primitives(pset)
    gp.add_ts_primitives(pset)
    gp.add_window_ephemeral(pset, window_name, 1, 5)
    return pset


def _samples(size=24, seed=31):
    generator = numpy.random.default_rng(seed)
    columns = [
        generator.normal(size=size),
        generator.normal(size=size) * 2.0,
        numpy.abs(generator.normal(size=size)),
    ]
    columns[1][3] = 0.0
    columns[2][7] = numpy.nan
    return tuple(columns)


def _matrix(columns):
    return numpy.ascontiguousarray(numpy.stack(columns, axis=1))


def _as_column(value, size):
    return numpy.broadcast_to(numpy.asarray(value, dtype=numpy.float64), (size,))


def _lower_trees(pset, count, seed=29):
    tools.rng.seed(seed)
    tapes = []
    while len(tapes) < count:
        tree = gp.PrimitiveTree(gp.gen_half_and_half(pset, 2, 4))
        try:
            tapes.append(gp.lower_tree(tree, pset))
        except ValueError:
            continue
    return tapes


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


def test_interpret_tapes_numba_matches_bind_tape():
    pset = _kit("BATCH_NUMBA_PARITY")
    columns = _samples()
    tapes = _lower_trees(pset, 12)
    actual = gp.interpret_tapes(tapes, _matrix(columns), backend="numba")

    for index, tape in enumerate(tapes):
        expected = gp.bind_tape(tape)(*columns)
        numpy.testing.assert_allclose(
            actual[index], expected, equal_nan=True, rtol=1e-9, atol=1e-12
        )


def test_interpret_tapes_numba_matches_the_default_backend():
    pset = _kit("BATCH_NUMBA_PYTHON")
    columns = _samples()
    tools.rng.seed(41)
    trees = []
    tapes = []
    while len(tapes) < 8:
        tree = gp.PrimitiveTree(gp.gen_half_and_half(pset, 2, 4))
        try:
            tapes.append(gp.lower_tree(tree, pset))
            trees.append(tree)
        except ValueError:
            continue
    actual = gp.interpret_tapes(tapes, _matrix(columns), backend="numba")

    for tree, row in zip(trees, actual, strict=True):
        expected = _as_column(gp.compile_tree(tree, pset)(*columns), 24)
        numpy.testing.assert_allclose(row, expected, equal_nan=True, rtol=1e-9, atol=1e-12)


def test_interpret_tapes_parallel_matches_serial():
    pset = _kit("BATCH_NUMBA_PARALLEL")
    columns = _samples(size=48)
    tapes = _lower_trees(pset, 10, seed=43)
    matrix = _matrix(columns)
    serial = gp.interpret_tapes(tapes, matrix, backend="numba")
    parallel = gp.interpret_tapes(tapes, matrix, backend="numba", parallel=True)

    numpy.testing.assert_allclose(parallel, serial, equal_nan=True, rtol=1e-12, atol=0.0)


def test_interpret_tapes_parallel_leaves_the_process_workspace():
    pset = _kit("BATCH_NUMBA_WORKSPACE")
    columns = _samples()
    tapes = _lower_trees(pset, 4, seed=47)
    gp.interpret_tapes(tapes[:1], _matrix(columns), backend="numba")
    held = numba_ops._workspace["stack"]

    gp.interpret_tapes(tapes, _matrix(columns), backend="numba", parallel=True)

    assert numba_ops._workspace["stack"] is held
    assert held.shape[1] == 24


def test_interpret_tapes_result_does_not_alias_the_matrix():
    pset = _kit("BATCH_NUMBA_COPY")
    columns = _samples()
    tape = gp.lower_tree(gp.PrimitiveTree([pset.mapping["first"]]), pset)
    matrix = _matrix(columns)

    actual = gp.interpret_tapes([tape], matrix, backend="numba")
    actual[0] = 0.0

    numpy.testing.assert_allclose(matrix[:, 0], columns[0])


def test_interpret_tapes_keeps_constant_pools_per_tape():
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    pset.add_terminal(2.0, gp.Array, "two")
    pset.add_terminal(5.0, gp.Array, "five")
    two = gp.lower_tree(gp.PrimitiveTree([pset.mapping["two"]]), pset)
    five = gp.lower_tree(gp.PrimitiveTree([pset.mapping["five"]]), pset)

    actual = gp.interpret_tapes([two, five], numpy.zeros((6, 3)), backend="numba")

    numpy.testing.assert_allclose(actual[0], 2.0)
    numpy.testing.assert_allclose(actual[1], 5.0)


def test_a_consumer_kernel_runs_in_a_mixed_batch():
    pset = _kit("BATCH_NUMBA_DISPATCH")
    pset.add_primitive(_triple, [gp.Array], gp.Array, "batch_numba_triple")
    gp.bind_numba_opcode("batch_numba_triple", TRIPLE)
    mapping = pset.mapping
    custom = gp.lower_tree(
        gp.PrimitiveTree([mapping["batch_numba_triple"], mapping["first"]]), pset
    )
    builtin = gp.lower_tree(gp.PrimitiveTree([mapping["vneg"], mapping["second"]]), pset)
    columns = _samples()

    for flag in (False, True):
        actual = gp.interpret_tapes(
            [custom, builtin],
            _matrix(columns),
            backend="numba",
            dispatch=_dispatch(),
            parallel=flag,
        )
        numpy.testing.assert_allclose(actual[0], 3.0 * columns[0], equal_nan=True)
        numpy.testing.assert_allclose(actual[1], -columns[1], equal_nan=True)


def test_a_consumer_kernel_without_a_dispatcher_is_rejected():
    pset = _kit("BATCH_NUMBA_ORPHAN")
    pset.add_primitive(_triple, [gp.Array], gp.Array, "batch_numba_orphan")
    gp.bind_numba_opcode("batch_numba_orphan", gp.USER_BASE + 82)
    tape = gp.lower_tree(
        gp.PrimitiveTree([pset.mapping["batch_numba_orphan"], pset.mapping["first"]]), pset
    )

    with pytest.raises(ValueError, match="Pass dispatch= to interpret_tapes"):
        gp.interpret_tapes([tape], numpy.zeros((4, 3)), backend="numba")


def test_interpret_tapes_rejects_a_set_without_arguments():
    pset = gp.PrimitiveSetTyped("MAIN", [], gp.Array)
    gp.add_numpy_primitives(pset)
    pset.add_terminal(2.0, gp.Array, "two")
    tape = gp.lower_tree(
        gp.PrimitiveTree([pset.mapping["vadd"], pset.mapping["two"], pset.mapping["two"]]), pset
    )

    with pytest.raises(ValueError, match="cannot size a result"):
        gp.interpret_tapes([tape], numpy.zeros((4, 0)), backend="numba")


def test_interpret_tapes_grows_the_shared_workspace_to_max_depth():
    pset = _kit("BATCH_NUMBA_DEPTH")
    columns = _samples()
    mapping = pset.mapping
    shallow = gp.lower_tree(gp.PrimitiveTree([mapping["vneg"], mapping["first"]]), pset)
    deep = gp.lower_tree(
        gp.PrimitiveTree(
            [
                mapping["vadd"],
                mapping["vmul"],
                mapping["first"],
                mapping["second"],
                mapping["third"],
            ]
        ),
        pset,
    )
    gp.interpret_tapes([shallow], _matrix(columns), backend="numba")
    gp.interpret_tapes([shallow, deep], _matrix(columns), backend="numba")

    assert numba_ops._workspace["stack"].shape[0] >= deep.depth + 1
    assert numba_ops._workspace["stack"].shape[1] == 24


def test_interpret_tapes_accepts_a_generator_of_tapes():
    pset = _kit("BATCH_NUMBA_GENERATOR")
    columns = _samples()
    tapes = _lower_trees(pset, 3, seed=53)

    actual = gp.interpret_tapes((tape for tape in tapes), _matrix(columns), backend="numba")

    for index, tape in enumerate(tapes):
        numpy.testing.assert_allclose(
            actual[index], gp.bind_tape(tape)(*columns), equal_nan=True, rtol=1e-9, atol=1e-12
        )


def test_a_serial_batch_does_not_compile_the_parallel_kernel():
    from deap_er.private.programming.numba import numba_batch

    pset = _kit("BATCH_NUMBA_SERIAL_ONLY")
    columns = _samples()
    tape = gp.lower_tree(gp.PrimitiveTree([pset.mapping["first"]]), pset)
    numba_batch._batch.clear()

    gp.interpret_tapes([tape], _matrix(columns), backend="numba")

    assert "many" in numba_batch._batch
    assert "many_parallel" not in numba_batch._batch

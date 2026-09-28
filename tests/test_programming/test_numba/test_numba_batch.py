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
from deap_er.private.programming.numba.numba_batch import compiled_batch_kernels, launch_kernels

pytestmark = pytest.mark.skipif(
    not gp.numba_available(), reason="the optional numba extra is not installed"
)

COLUMNS = ["first", "second", "third"]


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
    gp.bind_tape(tapes[0])(_matrix(columns))
    held = numba_ops.reserve(0, 24)[0].base

    gp.interpret_tapes(tapes, _matrix(columns), backend="numba", parallel=True)

    assert held is not None
    assert numba_ops.reserve(0, 24)[0].base is held
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


def test_interpret_tapes_rejects_a_set_without_arguments():
    pset = gp.PrimitiveSetTyped("MAIN", [], gp.Array)
    gp.add_numpy_primitives(pset)
    pset.add_terminal(2.0, gp.Array, "two")
    tape = gp.lower_tree(
        gp.PrimitiveTree([pset.mapping["vadd"], pset.mapping["two"], pset.mapping["two"]]), pset
    )

    with pytest.raises(ValueError, match="cannot size a result"):
        gp.interpret_tapes([tape], numpy.zeros((4, 0)), backend="numba")


def test_interpret_tapes_evaluates_tapes_of_different_depths():
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
    matrix = _matrix(columns)
    for parallel in (False, True):
        actual = gp.interpret_tapes([shallow, deep], matrix, backend="numba", parallel=parallel)
        for row, tape in zip(actual, (shallow, deep), strict=True):
            expected = gp.interpret_tape(tape, matrix)
            numpy.testing.assert_allclose(row, expected, equal_nan=True, rtol=1e-12)


def test_interpret_tapes_accepts_a_generator_of_tapes():
    pset = _kit("BATCH_NUMBA_GENERATOR")
    columns = _samples()
    tapes = _lower_trees(pset, 3, seed=53)

    actual = gp.interpret_tapes((tape for tape in tapes), _matrix(columns), backend="numba")

    for index, tape in enumerate(tapes):
        numpy.testing.assert_allclose(
            actual[index], gp.bind_tape(tape)(*columns), equal_nan=True, rtol=1e-9, atol=1e-12
        )


def test_a_serial_builtin_batch_compiles_no_batch_kernel():
    pset = _kit("BATCH_NUMBA_SERIAL_ONLY")
    columns = _samples()
    tape = gp.lower_tree(gp.PrimitiveTree([pset.mapping["first"]]), pset)
    before = compiled_batch_kernels()

    gp.interpret_tapes([tape], _matrix(columns), backend="numba")

    assert compiled_batch_kernels() == before


def _hand_tape(opcodes, operands, depth=2, constants=()):
    return gp.Tape(
        opcodes=numpy.array(opcodes, dtype=numpy.int32),
        operands=numpy.array(operands, dtype=numpy.int32),
        constants=numpy.array(constants, dtype=numpy.float64),
        columns=1,
        depth=depth,
        fill=1.0,
    )


MALFORMED = {
    "window_below_one": (_hand_tape([gp.Opcode.COL_LOAD, gp.Opcode.DELAY], [0, -2]), "operand -2"),
    "zero_window": (_hand_tape([gp.Opcode.COL_LOAD, gp.Opcode.ROLL_SUM], [0, 0]), "operand 0"),
    "underflow": (_hand_tape([gp.Opcode.COL_LOAD, gp.Opcode.ADD], [0, -1]), "underflows"),
    "column": (_hand_tape([gp.Opcode.COL_LOAD], [5]), "operand 5"),
    "constant": (_hand_tape([gp.Opcode.CONST], [3]), "operand 3"),
    "depth": (
        _hand_tape([gp.Opcode.COL_LOAD] * 3 + [gp.Opcode.ADD] * 2, [0, 0, 0, -1, -1], depth=1),
        "above its declared",
    ),
}


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_the_compiled_entry_points_reject_a_malformed_tape(case):
    tape, message = MALFORMED[case]
    matrix = numpy.arange(6.0).reshape(6, 1)

    with pytest.raises(ValueError, match=message):
        gp.bind_tape(tape)
    with pytest.raises(ValueError, match=message):
        launch_kernels([tape], matrix, None, False)

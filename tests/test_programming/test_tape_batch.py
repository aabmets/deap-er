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

COLUMNS = ["first", "second", "third"]


def _kit(window_name, fill=None):
    pset = gp.make_column_pset(COLUMNS)
    kwargs = {} if fill is None else {"fill": fill}
    gp.add_numpy_primitives(pset, **kwargs)
    gp.add_window_primitives(pset)
    gp.add_window_ephemeral(pset, window_name, 1, 5)
    return pset


def _samples(size=24, seed=17):
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


def _lower_trees(pset, count, seed=23):
    tools.rng.seed(seed)
    tapes = []
    while len(tapes) < count:
        tree = gp.PrimitiveTree(gp.gen_half_and_half(pset, 2, 4))
        try:
            tapes.append(gp.lower_tree(tree, pset))
        except ValueError:
            continue
    return tapes


def test_interpret_tapes_matches_interpret_tape_on_random_trees():
    pset = _kit("BATCH_OPCODE_PARITY")
    columns = _samples()
    tapes = _lower_trees(pset, 12)
    actual = gp.interpret_tapes(tapes, _matrix(columns))

    assert actual.shape == (12, 24)
    assert actual.dtype == numpy.float64
    assert actual.flags["C_CONTIGUOUS"]
    for index, tape in enumerate(tapes):
        expected = _as_column(gp.interpret_tape(tape, columns), 24)
        numpy.testing.assert_allclose(actual[index], expected, equal_nan=True)


def test_interpret_tapes_broadcasts_a_constant_and_copies_a_column():
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    pset.add_terminal(2.5, gp.Array, "two_half")
    constant = gp.lower_tree(gp.PrimitiveTree([pset.mapping["two_half"]]), pset)
    column = gp.lower_tree(gp.PrimitiveTree([pset.mapping["first"]]), pset)
    columns = _samples()
    matrix = _matrix(columns)

    actual = gp.interpret_tapes([constant, column], matrix)
    first = actual[1].copy()
    actual[1] = 0.0

    numpy.testing.assert_allclose(actual[0], 2.5)
    numpy.testing.assert_allclose(first, columns[0])
    numpy.testing.assert_allclose(matrix[:, 0], columns[0])


def test_interpret_tapes_honors_per_tape_fill():
    pset_a = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset_a, fill=3.0)
    pset_b = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset_b, fill=-1.0)
    mapping = pset_a.mapping
    tree = gp.PrimitiveTree([mapping["vdiv"], mapping["first"], mapping["second"]])
    tape_a = gp.lower_tree(tree, pset_a)
    tape_b = gp.lower_tree(tree, pset_b)
    columns = (numpy.array([1.0, 1.0]), numpy.array([0.0, 1.0]), numpy.array([1.0, 1.0]))

    actual = gp.interpret_tapes([tape_a, tape_b], _matrix(columns))

    numpy.testing.assert_allclose(actual[0], [3.0, 1.0])
    numpy.testing.assert_allclose(actual[1], [-1.0, 1.0])


def test_interpret_tapes_rejects_a_sequence_of_columns():
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    tape = gp.lower_tree(gp.PrimitiveTree([pset.mapping["first"]]), pset)

    with pytest.raises(ValueError, match="not a sequence of columns"):
        gp.interpret_tapes([tape], _samples())


def test_interpret_tapes_rejects_a_one_dimensional_matrix():
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    tape = gp.lower_tree(gp.PrimitiveTree([pset.mapping["first"]]), pset)

    with pytest.raises(ValueError, match="ndim=1"):
        gp.interpret_tapes([tape], numpy.zeros(4))


def test_interpret_tapes_rejects_a_wrong_column_count():
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    tape = gp.lower_tree(gp.PrimitiveTree([pset.mapping["first"]]), pset)

    with pytest.raises(ValueError, match="expects 3 columns"):
        gp.interpret_tapes([tape], numpy.zeros((4, 2)))


def test_interpret_tapes_rejects_an_unknown_backend():
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    tape = gp.lower_tree(gp.PrimitiveTree([pset.mapping["first"]]), pset)

    with pytest.raises(ValueError, match="Unknown compile backend"):
        gp.interpret_tapes([tape], numpy.zeros((4, 3)), backend="rust")


def test_interpret_tapes_rejects_parallel_on_the_opcode_backend():
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    tape = gp.lower_tree(gp.PrimitiveTree([pset.mapping["first"]]), pset)

    with pytest.raises(ValueError, match="requires backend='numba'"):
        gp.interpret_tapes([tape], numpy.zeros((4, 3)), parallel=True)


def test_interpret_tapes_rejects_a_consumer_opcode_on_the_opcode_backend():
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    pset.add_primitive(numpy.tanh, [gp.Array], gp.Array, "batch_local_kernel")
    gp.bind_numba_opcode("batch_local_kernel", gp.USER_BASE + 71)
    tree = gp.PrimitiveTree([pset.mapping["batch_local_kernel"], pset.mapping["first"]])
    tape = gp.lower_tree(tree, pset)

    with pytest.raises(ValueError, match="consumer kernel"):
        gp.interpret_tapes([tape], numpy.zeros((4, 3)))


def test_interpret_tapes_returns_an_empty_batch():
    actual = gp.interpret_tapes([], numpy.zeros((8, 3)))

    assert actual.shape == (0, 8)
    assert actual.dtype == numpy.float64


def test_interpret_tapes_returns_zero_rows():
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    tape = gp.lower_tree(gp.PrimitiveTree([pset.mapping["first"]]), pset)

    actual = gp.interpret_tapes([tape], numpy.zeros((0, 3)))

    assert actual.shape == (1, 0)


def test_interpret_tapes_runs_tapes_with_empty_constant_pools():
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    tapes = [
        gp.lower_tree(gp.PrimitiveTree([pset.mapping["first"]]), pset),
        gp.lower_tree(gp.PrimitiveTree([pset.mapping["second"]]), pset),
    ]
    columns = _samples()

    actual = gp.interpret_tapes(tapes, _matrix(columns))

    numpy.testing.assert_allclose(actual[0], columns[0])
    numpy.testing.assert_allclose(actual[1], columns[1])

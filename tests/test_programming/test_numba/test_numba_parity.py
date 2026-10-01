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

pytestmark = pytest.mark.skipif(
    not gp.numba_available(), reason="the optional numba extra is not installed"
)

# The tolerance documented on ``interpret_tapes`` for short series.
RTOL = 1e-9
ATOL = 1e-12


def _tapes(count=6, seed=67):
    pset = gp.make_column_pset(["first", "second", "third"])
    gp.add_numpy_primitives(pset)
    gp.add_window_primitives(pset)
    gp.add_pair_window_primitives(pset)
    gp.add_ts_primitives(pset)
    gp.add_window_ephemeral(pset, "NUMBA_PARITY_WINDOW", 1, 5)
    tools.rng.seed(seed)
    tapes = []
    while len(tapes) < count:
        tree = gp.PrimitiveTree(gp.gen_half_and_half(pset, 2, 4))
        try:
            tapes.append(gp.lower_tree(tree, pset))
        except ValueError:
            continue
    return tapes


def _matrix(rows=64, seed=71):
    generator = numpy.random.default_rng(seed)
    matrix = generator.normal(size=(rows, 3))
    matrix[:, 2] = numpy.abs(matrix[:, 2])
    matrix[3, 1] = 0.0
    matrix[7, 2] = numpy.nan
    return numpy.ascontiguousarray(matrix)


@pytest.fixture
def two_threads():
    # Optional extra: numba may be absent, in which case the module is skipped.
    import numba

    previous = numba.get_num_threads()
    try:
        numba.set_num_threads(2)
    except ValueError:
        pytest.skip("numba has a single thread, so parallel=True runs the serial kernel")
    try:
        yield
    finally:
        numba.set_num_threads(previous)


def _assert_close(actual, expected):
    numpy.testing.assert_array_equal(numpy.isnan(actual), numpy.isnan(expected))
    numpy.testing.assert_allclose(actual, expected, equal_nan=True, rtol=RTOL, atol=ATOL)


@pytest.mark.usefixtures("two_threads")
def test_parallel_serial_and_opcode_paths_agree_within_tolerance():
    tapes = _tapes()
    matrix = _matrix()

    parallel = gp.interpret_tapes(tapes, matrix, backend="numba", parallel=True)
    serial = gp.interpret_tapes(tapes, matrix, backend="numba")
    opcode = gp.interpret_tapes(tapes, matrix, backend="opcode")

    _assert_close(parallel, serial)
    _assert_close(serial, opcode)
    _assert_close(parallel, opcode)

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
from deap_er.private.programming import tape_cse
from deap_er.private.programming.numba import numba_batch
from tests.harness.numba_dispatch import BATCH_TRIPLE, consumer_dispatch

pytestmark = pytest.mark.skipif(
    not gp.numba_available(), reason="the optional numba extra is not installed"
)

COLUMNS = ["first", "second", "third"]
TRIPLE = BATCH_TRIPLE


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
    dispatch = consumer_dispatch()

    for flag in (False, True):
        actual = gp.interpret_tapes(
            [custom, builtin],
            _matrix(columns),
            backend="numba",
            dispatch=dispatch,
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


def test_serial_numba_without_consumer_routes_through_cse(monkeypatch):
    pset = _kit("BATCH_NUMBA_CSE_ROUTE")
    columns = _samples()
    tapes = _lower_trees(pset, 4, seed=59)
    matrix = _matrix(columns)
    routed = {"cse": False}
    original = tape_cse.run_opcode_cse

    def tracking(tapes, matrix):
        routed["cse"] = True
        return original(tapes, matrix)

    monkeypatch.setattr(
        "deap_er.private.programming.numba.numba_batch.run_opcode_cse",
        tracking,
    )
    actual = gp.interpret_tapes(tapes, matrix, backend="numba")
    expected = gp.interpret_tapes(tapes, matrix, backend="opcode")
    assert routed["cse"] is True
    numpy.testing.assert_allclose(actual, expected, equal_nan=True)


def test_serial_numba_with_consumer_skips_cse(monkeypatch):
    pset = _kit("BATCH_NUMBA_CSE_SKIP")
    pset.add_primitive(_triple, [gp.Array], gp.Array, "batch_numba_cse_skip")
    gp.bind_numba_opcode("batch_numba_cse_skip", TRIPLE)
    mapping = pset.mapping
    tape = gp.lower_tree(
        gp.PrimitiveTree([mapping["batch_numba_cse_skip"], mapping["first"]]), pset
    )
    columns = _samples()
    routed = {"cse": False, "many": 0}
    original_cse = tape_cse.run_opcode_cse
    original_serial = numba_batch._serial_kernel

    def tracking_cse(tapes, matrix):
        routed["cse"] = True
        return original_cse(tapes, matrix)

    def tracking_serial():
        run, idle, many = original_serial()

        def wrapped_many(*args, **kwargs):
            routed["many"] += 1
            return many(*args, **kwargs)

        return run, idle, wrapped_many

    monkeypatch.setattr(numba_batch, "run_opcode_cse", tracking_cse)
    monkeypatch.setattr(numba_batch, "_serial_kernel", tracking_serial)
    actual = gp.interpret_tapes(
        [tape],
        _matrix(columns),
        backend="numba",
        dispatch=consumer_dispatch(),
    )
    assert routed["cse"] is False
    assert routed["many"] == 1
    numpy.testing.assert_allclose(actual[0], 3.0 * columns[0], equal_nan=True)

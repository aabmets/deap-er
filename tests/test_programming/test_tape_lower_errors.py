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

COLUMNS = ["first", "second", "third"]


def _kit(window_name):
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    gp.add_window_primitives(pset)
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


def test_lowering_rejects_a_primitive_without_an_opcode():
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    pset.add_primitive(numpy.tanh, [gp.Array], gp.Array, "vtanh")
    tree = gp.PrimitiveTree([pset.mapping["vtanh"], pset.mapping["first"]])

    with pytest.raises(ValueError, match="no builtin opcode"):
        gp.lower_tree(tree, pset)


def test_lowering_rejects_a_window_that_is_not_a_leaf():
    pset = _kit("OPCODES_COMPUTED_WINDOW")
    pset.add_primitive(min, [gp.Window, gp.Window], gp.Window, "wmin")
    mapping = pset.mapping
    window = pset.terminals[gp.Window][0]
    tree = gp.PrimitiveTree(
        [mapping["delay"], mapping["first"], mapping["wmin"], window(), window()]
    )

    with pytest.raises(ValueError, match="must be a leaf"):
        gp.lower_tree(tree, pset)


def test_lowering_rejects_a_terminal_that_is_not_a_column_or_a_number():
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    pset.add_terminal(object(), gp.Array, "opaque")
    tree = gp.PrimitiveTree([pset.mapping["opaque"]])

    with pytest.raises(ValueError, match="cannot be lowered"):
        gp.lower_tree(tree, pset)


def test_lowering_rejects_an_empty_expression():
    pset = gp.make_column_pset(COLUMNS)

    empty = gp.PrimitiveTree([])
    with pytest.raises(ValueError, match="empty expression"):
        gp.lower_tree(empty, pset)


def test_lowering_rejects_a_tree_with_unreachable_nodes():
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    tree = gp.PrimitiveTree([pset.mapping["first"], pset.mapping["second"]])

    with pytest.raises(ValueError, match="not reachable"):
        gp.lower_tree(tree, pset)


def test_lowering_rejects_a_tree_that_underflows_the_stack():
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    tree = gp.PrimitiveTree([pset.mapping["vadd"], pset.mapping["first"]])

    with pytest.raises(ValueError, match="underflow"):
        gp.lower_tree(tree, pset)


def test_interpreting_rejects_a_wrong_column_count():
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    tape = gp.lower_tree(gp.PrimitiveTree([pset.mapping["first"]]), pset)

    with pytest.raises(ValueError, match="expects 3 columns"):
        gp.interpret_tape(tape, (numpy.zeros(4),))


def test_consumer_opcodes_are_rejected_by_the_opcode_backend():
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    pset.add_primitive(numpy.tanh, [gp.Array], gp.Array, "opcodes_local_kernel")
    gp.bind_numba_opcode("opcodes_local_kernel", gp.USER_BASE + 41)
    tree = gp.PrimitiveTree([pset.mapping["opcodes_local_kernel"], pset.mapping["first"]])

    assert gp.numba_opcodes()["opcodes_local_kernel"] == gp.USER_BASE + 41
    with pytest.raises(ValueError, match="consumer kernel"):
        gp.compile_tree(tree, pset, backend="opcode")


def test_interpreting_rejects_an_opcode_it_cannot_run():
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)
    tape = gp.lower_tree(gp.PrimitiveTree([pset.mapping["first"]]), pset)
    tape.opcodes[0] = gp.USER_BASE + 3

    columns = _samples()
    with pytest.raises(ValueError, match="no Python implementation"):
        gp.interpret_tape(tape, columns)


@pytest.mark.parametrize(
    ("name", "opcode", "reason"),
    [
        ("whatever", 7, "at least"),
        ("vadd", gp.USER_BASE + 1, "builtin opcode"),
    ],
)
def test_binding_a_consumer_opcode_is_validated(name, opcode, reason):
    with pytest.raises(ValueError, match=reason):
        gp.bind_numba_opcode(name, opcode)


def test_binding_the_same_name_twice_is_idempotent():
    gp.bind_numba_opcode("opcodes_stable_kernel", gp.USER_BASE + 42)
    gp.bind_numba_opcode("opcodes_stable_kernel", gp.USER_BASE + 42)

    with pytest.raises(ValueError, match="already bound"):
        gp.bind_numba_opcode("opcodes_stable_kernel", gp.USER_BASE + 43)


def test_compile_tree_rejects_an_unknown_backend():
    pset = gp.make_column_pset(COLUMNS)
    gp.add_numpy_primitives(pset)

    tree = gp.PrimitiveTree([pset.mapping["first"]])
    with pytest.raises(ValueError, match="Unknown compile backend"):
        gp.compile_tree(tree, pset, backend="rust")


def test_builtin_opcodes_stay_below_the_consumer_range():
    assert all(opcode < gp.USER_BASE for opcode in gp.BUILTIN_OPCODES.values())
    assert len(set(gp.BUILTIN_OPCODES.values())) == len(gp.BUILTIN_OPCODES)

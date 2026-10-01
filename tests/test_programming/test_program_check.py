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
import operator
from typing import Any

import numpy
import pytest
from deap_er import gp, tools


def _ephemeral(pset, slot, name):
    return next(item for item in pset.terminals[slot] if getattr(item, "__name__", "") == name)


def _kit(window=None):
    pset = gp.make_column_pset(["a", "b"])
    gp.add_numpy_primitives(pset)
    gp.add_window_primitives(pset, ema=False)
    if window is not None:
        gp.add_window_ephemeral(pset, *window)
    return pset


def test_program_error_is_a_value_and_type_error():
    assert issubclass(gp.ProgramError, ValueError)
    assert issubclass(gp.ProgramError, TypeError)


@pytest.mark.parametrize("text", ["vgt(a, b)", "True"])
def test_from_string_refuses_a_root_of_the_wrong_type(text):
    with pytest.raises(gp.ProgramError, match="at the root") as info:
        gp.PrimitiveTree.from_string(text, _kit())
    assert "Mask" in str(info.value)
    assert "Array" in str(info.value)


def test_from_string_keeps_accepting_literal_roots_of_untyped_sets():
    pset = gp.PrimitiveSet("main", 1)
    pset.add_primitive(operator.add, 2)

    assert gp.PrimitiveTree.from_string("2", pset)[0].value == 2
    assert gp.PrimitiveTree.from_string("ARG0", pset)[0].name == "ARG0"


@pytest.mark.parametrize("text", ["", "  \n", "vadd(a,", "vadd(a, b) a", "vadd(a, foo)"])
def test_from_string_raises_program_error_for_every_parse_failure(text):
    with pytest.raises(gp.ProgramError):
        gp.PrimitiveTree.from_string(text, _kit())


@pytest.mark.parametrize("expr", [gp.PrimitiveTree([]), "", " "])
def test_compile_tree_refuses_an_empty_expression(expr):
    with pytest.raises(gp.ProgramError, match="empty"):
        gp.compile_tree(expr, _kit())


@pytest.mark.parametrize("window", ["99999999999999999999999", "2147483648"])
def test_from_string_refuses_a_window_past_the_operand_width(window):
    with pytest.raises(gp.ProgramError, match="'delay'") as info:
        gp.PrimitiveTree.from_string(f"delay(a, {window})", _kit())
    assert window in str(info.value)


@pytest.mark.parametrize("window", [2**31, 10**23])
def test_lowering_refuses_a_built_window_past_the_operand_width(window):
    pset = _kit()
    tree = gp.PrimitiveTree(
        [pset.mapping["delay"], pset.mapping["a"], gp.Terminal(window, False, gp.Window)]
    )
    with pytest.raises(gp.ProgramError, match="no larger than 2147483647"):
        gp.lower_tree(tree, pset)


@pytest.mark.parametrize("window", ["0", "-2"])
def test_from_string_refuses_a_window_below_one(window):
    with pytest.raises(gp.ProgramError, match="'rolling_mean'") as info:
        gp.PrimitiveTree.from_string(f"rolling_mean(a, {window})", _kit())
    assert f"window argument {window} " in str(info.value)


@pytest.mark.parametrize("window", [0, -2, 2.5])
def test_compile_tree_refuses_a_built_window_below_one(window):
    pset = _kit()
    tree = gp.PrimitiveTree(
        [pset.mapping["rolling_mean"], pset.mapping["a"], gp.Terminal(window, False, gp.Window)]
    )
    with pytest.raises(gp.ProgramError, match="'rolling_mean'"):
        gp.compile_tree(tree, pset)


def test_compile_tree_refuses_a_terminal_whose_text_would_not_parse_back():
    pset = _kit()
    parsed = gp.PrimitiveTree.from_string("vadd(a, b)", pset)
    tree = gp.PrimitiveTree([*parsed[:2], gp.Terminal(0.5, False, gp.Array)])

    with pytest.raises(gp.ProgramError, match="does not match"):
        gp.compile_tree(tree, pset, backend="python")
    with pytest.raises(gp.ProgramError, match="does not match"):
        gp.lower_tree(tree, pset)


def test_a_number_in_a_slot_with_an_ephemeral_round_trips():
    pset = _kit()
    pset.add_ephemeral_constant("PROGRAM_CHECK_ARRAY_CONST", lambda: 0.25, gp.Array)
    ephemeral = _ephemeral(pset, gp.Array, "PROGRAM_CHECK_ARRAY_CONST")
    tree = gp.PrimitiveTree([pset.mapping["vadd"], pset.mapping["a"], ephemeral()])

    restored = gp.PrimitiveTree.from_string(str(tree), pset)

    assert restored == tree
    assert type(restored[-1]) is ephemeral
    numpy.testing.assert_allclose(gp.compile_tree(restored, pset)(numpy.ones(2), None), 1.25)


def test_generated_trees_round_trip_through_text():
    tools.rng.seed(5)
    pset = _kit(("PROGRAM_CHECK_ROUND_TRIP", 2, 9))
    for _ in range(200):
        tree = gp.PrimitiveTree(gp.gen_half_and_half(pset, 1, 4))
        gp.compile_tree(tree, pset)
        assert gp.PrimitiveTree.from_string(str(tree), pset) == tree


def test_from_string_restores_a_window_literal_as_the_window_ephemeral():
    tools.rng.seed(3)
    pset = _kit(("PROGRAM_CHECK_WINDOW", 2, 9))
    tree = gp.PrimitiveTree.from_string("rolling_mean(a, 5)", pset)
    window = tree[-1]

    assert isinstance(window, gp.Ephemeral)
    assert type(window) is _ephemeral(pset, gp.Window, "PROGRAM_CHECK_WINDOW")
    assert window.value == 5
    assert str(tree) == "rolling_mean(a, 5)"

    individual: Any = tree
    seen = set()
    for _ in range(50):
        (individual,) = gp.mut_ephemeral(individual, mode="all")
        seen.add(individual[-1].value)
    assert len(seen) > 1
    assert seen <= set(range(2, 10))


def test_from_string_picks_the_window_ephemeral_whose_bounds_hold_the_value():
    pset = _kit(("PROGRAM_CHECK_SHORT", 1, 4))
    gp.add_window_ephemeral(pset, "PROGRAM_CHECK_LONG", 10, 20)

    short = gp.PrimitiveTree.from_string("delay(a, 3)", pset)[-1]
    long = gp.PrimitiveTree.from_string("delay(a, 15)", pset)[-1]
    outside = gp.PrimitiveTree.from_string("delay(a, 7)", pset)[-1]

    assert type(short) is _ephemeral(pset, gp.Window, "PROGRAM_CHECK_SHORT")
    assert type(long) is _ephemeral(pset, gp.Window, "PROGRAM_CHECK_LONG")
    assert type(outside) is _ephemeral(pset, gp.Window, "PROGRAM_CHECK_SHORT")
    assert outside.value == 7


def test_a_window_literal_stays_a_terminal_without_a_window_ephemeral():
    window = gp.PrimitiveTree.from_string("delay(a, 3)", _kit())[-1]

    assert type(window) is gp.Terminal
    assert window.ret is gp.Window

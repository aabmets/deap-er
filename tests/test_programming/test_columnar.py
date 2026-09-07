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


def test_make_column_pset_keeps_column_order():
    pset = gp.make_column_pset(["first", "second", "third"])

    assert pset.arguments == ["first", "second", "third"]
    assert pset.ins == [gp.Array, gp.Array, gp.Array]
    assert pset.ret is gp.Array
    assert pset.name == "MAIN"


def test_make_column_pset_accepts_a_custom_name():
    assert gp.make_column_pset(["only"], name="OTHER").name == "OTHER"


def test_compiled_tree_takes_columns_in_declaration_order():
    pset = gp.make_column_pset(["left", "right"])
    gp.add_numpy_primitives(pset)
    tree = gp.PrimitiveTree([pset.mapping["vsub"], pset.mapping["left"], pset.mapping["right"]])

    func = gp.compile_tree(tree, pset)
    result = func(numpy.array([5.0, 6.0]), numpy.array([1.0, 2.0]))

    assert str(tree) == "vsub(left, right)"
    numpy.testing.assert_allclose(result, [4.0, 4.0])


def test_compiled_tree_returns_a_float_column():
    pset = gp.make_column_pset(["value"])
    gp.add_numpy_primitives(pset)
    tree = gp.PrimitiveTree([pset.mapping["vabs"], pset.mapping["value"]])

    result = gp.compile_tree(tree, pset)(numpy.linspace(-1.0, 1.0, 32))

    assert result.ndim == 1
    assert result.shape == (32,)
    assert result.dtype == numpy.float64


@pytest.mark.parametrize(
    ("names", "reason"),
    [
        ([], "At least one column"),
        (["ok", "not an identifier"], "identifier"),
        (["ok", "9lives"], "identifier"),
        (["ok", 7], "identifier"),
        (["ok", "class"], "keyword"),
        (["ok", "match"], "keyword"),
        (["ok", "ARG1"], "argument prefix"),
        (["ARG0"], "argument prefix"),
        (["same", "same"], "not unique"),
    ],
)
def test_make_column_pset_rejects_unusable_names(names, reason):
    with pytest.raises(ValueError, match=reason):
        gp.make_column_pset(names)


def test_type_tags_are_distinct_classes():
    tags = (gp.Array, gp.Mask, gp.Window)

    assert all(isinstance(tag, type) for tag in tags)
    assert len({*tags}) == 3
    assert not any(issubclass(one, other) for one in tags for other in tags if one is not other)


def _window_pset() -> gp.PrimitiveSetTyped:
    pset = gp.make_column_pset(["price"])
    gp.add_numpy_primitives(pset)
    gp.add_window_primitives(pset)
    return pset


def test_from_string_accepts_a_window_integer_leaf():
    pset = _window_pset()
    tree = gp.PrimitiveTree.from_string("rolling_mean(price, 3)", pset)

    assert tree[-1].ret is gp.Window
    assert tree[-1].value == 3
    assert str(tree) == "rolling_mean(price, 3)"


def test_stringified_window_tree_round_trips():
    pset = _window_pset()
    original = gp.PrimitiveTree(
        [pset.mapping["rolling_mean"], pset.mapping["price"], gp.Terminal(3, False, gp.Window)]
    )

    restored = gp.PrimitiveTree.from_string(str(original), pset)

    assert str(restored) == str(original)
    assert restored[-1].ret is gp.Window
    assert restored[-1].value == 3


def test_opcode_backend_compiles_a_stringified_window_tree():
    pset = _window_pset()
    original = gp.PrimitiveTree(
        [pset.mapping["delay"], pset.mapping["price"], gp.Terminal(2, False, gp.Window)]
    )
    text = str(original)
    column = numpy.arange(8, dtype=numpy.float64)
    gp.clear_compile_cache()

    expected = gp.compile_tree(text, pset)(column)
    actual = gp.compile_tree(text, pset, backend="opcode")(column)

    numpy.testing.assert_allclose(actual, expected, equal_nan=True)
    numpy.testing.assert_allclose(actual, gp.delay(column, 2), equal_nan=True)


def test_from_string_rejects_a_non_integer_window_literal():
    pset = _window_pset()

    with pytest.raises(TypeError, match="does not match"):
        gp.PrimitiveTree.from_string("rolling_mean(price, 3.5)", pset)
    with pytest.raises(TypeError, match="return type"):
        gp.PrimitiveTree.from_string("rolling_mean(price, True)", pset)

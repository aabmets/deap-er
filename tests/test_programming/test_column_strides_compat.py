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

NAMES = ["fast", "slow", "slower"]


def _pset(strides=None):
    return gp.columnar_pset(
        NAMES, window=(1, 5), window_name="STRIDES_COMPAT_WINDOW", ts=True, strides=strides
    )


def _trees(pset, count=40, seed=29):
    tools.rng.seed(seed)
    return [gp.PrimitiveTree(gp.gen_half_and_half(pset, 2, 4)) for _ in range(count)]


def _matrix(rows=96, seed=31):
    generator = numpy.random.default_rng(seed)
    fast = generator.normal(size=rows)
    slow = numpy.repeat(generator.normal(size=rows // 4 + 1), 4)[1 : rows + 1]
    slower = numpy.abs(numpy.repeat(generator.normal(size=rows // 6 + 1), 6)[:rows])
    return numpy.ascontiguousarray(numpy.column_stack([fast, slow, slower]))


def _backends():
    return ("python", "opcode", "numba") if gp.numba_available() else ("python", "opcode")


def test_unit_strides_store_nothing():
    assert not hasattr(_pset({"slow": 1, "slower": 1}), "column_strides")
    assert not hasattr(_pset({}), "column_strides")
    assert _pset({"slow": 4}).column_strides == (1, 4, 1)


def test_unit_strides_lower_and_evaluate_bit_identically():
    plain, unit = _pset(), _pset({"slow": 1})
    matrix = _matrix()
    columns = tuple(matrix.T)
    for tree in _trees(plain):
        text = str(tree)
        first, second = gp.lower_tree(text, plain), gp.lower_tree(text, unit)
        numpy.testing.assert_array_equal(first.opcodes, second.opcodes)
        numpy.testing.assert_array_equal(first.operands, second.operands)
        for backend in _backends():
            expected = gp.compile_tree(
                gp.PrimitiveTree.from_string(text, plain), plain, backend=backend
            )
            actual = gp.compile_tree(
                gp.PrimitiveTree.from_string(text, unit), unit, backend=backend
            )
            numpy.testing.assert_array_equal(
                numpy.asarray(actual(*columns)), numpy.asarray(expected(*columns))
            )


def test_strided_programs_agree_across_backends():
    pset = _pset({"slow": 4, "slower": 6})
    matrix = _matrix()
    columns = tuple(matrix.T)
    trees = _trees(pset, seed=37)
    tapes = [gp.lower_tree(tree, pset) for tree in trees]
    batches = {"opcode": gp.interpret_tapes(tapes, matrix)}
    if gp.numba_available():
        batches["numba"] = gp.interpret_tapes(tapes, matrix, backend="numba")
    for slot, tree in enumerate(trees):
        python = numpy.broadcast_to(gp.compile_tree(tree, pset)(*columns), (matrix.shape[0],))
        for batch in batches.values():
            numpy.testing.assert_array_equal(numpy.isnan(batch[slot]), numpy.isnan(python))
            numpy.testing.assert_allclose(
                batch[slot], python, rtol=1e-9, atol=1e-12, equal_nan=True
            )


def _window_set(strides=None):
    # Window primitives are module-level functions, so both sets share
    # every context object and differ only in strides.
    pset = gp.make_column_pset(NAMES, strides=strides)
    gp.add_window_primitives(pset)
    return pset


def test_compile_cache_tells_strides_apart():
    plain, strided = _window_set(), _window_set({"slow": 4})
    columns = tuple(_matrix().T)
    for backend in _backends():
        results = [
            numpy.asarray(
                gp.compile_tree(
                    gp.PrimitiveTree.from_string("delay(slow, 1)", pset), pset, backend=backend
                )(*columns)
            )
            for pset in (plain, strided)
        ]
        numpy.testing.assert_array_equal(results[0][1:], columns[1][:-1])
        numpy.testing.assert_array_equal(results[1][4:], columns[1][:-4])


def test_python_backend_scales_source_text_and_deep_trees():
    pset = _pset({"slow": 4})
    columns = tuple(_matrix().T)
    text = "delay(slow, 2)"
    from_text = numpy.asarray(gp.compile_tree(text, pset)(*columns))
    numpy.testing.assert_array_equal(from_text[8:], columns[1][:-8])
    deep = "delay(slow, 1)"
    for _ in range(70):
        deep = f"vadd({deep}, fast)"
    tree = gp.PrimitiveTree.from_string(deep, pset)
    python = gp.compile_tree(tree, pset)(*columns)
    opcode = gp.compile_tree(tree, pset, backend="opcode")(*columns)
    numpy.testing.assert_allclose(python, opcode, equal_nan=True)


@pytest.mark.parametrize("backend", ["python", "opcode"])
def test_promoted_primitives_take_the_stride_of_their_arguments(backend):
    pset = _pset({"slow": 4})
    name = gp.promote_subtree(pset, gp.PrimitiveTree.from_string("vmul(fast, slower)", pset))
    columns = tuple(_matrix().T)
    tree = gp.PrimitiveTree.from_string(f"diff({name}(slow, slow), 1)", pset)
    result = numpy.asarray(gp.compile_tree(tree, pset, backend=backend)(*columns))
    square = columns[1] * columns[1]
    numpy.testing.assert_allclose(result[4:], square[4:] - square[:-4])
    assert gp.tape_lookback(gp.lower_tree(tree, pset)) == 4


def test_evaluate_columnar_uses_the_strides():
    pset = _pset({"slow": 4})
    matrix = _matrix()
    target = numpy.concatenate([numpy.full(4, numpy.nan), matrix[:-4, 1]])
    tree = gp.PrimitiveTree.from_string("delay(slow, 1)", pset)
    assert gp.evaluate_columnar([tree], pset, matrix, target) == [(0.0,)]

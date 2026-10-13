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
import tracemalloc
from typing import Any

import numpy
import pytest
from deap_er import Fitness, creator, gp, tools
from deap_er.private.programming import columnar_setup, cx_semantic_one_point, suffix_rescore
from deap_er.private.programming.matrix_pack import as_matrix
from deap_er.private.programming.tape_interval import bounds_from_matrix
from tests.test_programming.precision_cases import (
    GOLDEN_DIGEST,
    GOLDEN_TEXTS,
    MATH_TEXTS,
    digest,
    precision_matrix,
    precision_pset,
    random_tapes,
    text_tapes,
)

# Typed as Any so the type checker lets the bad value reach the check.
BAD_PRECISION: Any = "float16"
SEMANTIC_FIT = "PRECISION_SEMANTIC_FIT"
SEMANTIC_IND = "PRECISION_SEMANTIC_IND"


def _all_tapes(pset):
    return text_tapes(pset, GOLDEN_TEXTS + MATH_TEXTS) + random_tapes(pset, 40)


def _spy(monkeypatch, module):
    """Record each matrix the module hands to interpret_tapes."""
    seen = []
    original = module.interpret_tapes

    def record(tapes, matrix, **kwargs):
        seen.append(matrix)
        return original(tapes, matrix, **kwargs)

    monkeypatch.setattr(module, "interpret_tapes", record)
    return seen


def _peak_bytes(call):
    tracemalloc.start()
    try:
        call()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("cast", [False, True])
def test_the_default_precision_reproduces_the_3_2_9_results(cast):
    pset = precision_pset()
    m32 = precision_matrix()
    matrix = m32 if cast else m32.astype(numpy.float64)
    results = gp.interpret_tapes(text_tapes(pset, GOLDEN_TEXTS), matrix)
    assert digest(results) == GOLDEN_DIGEST
    explicit = gp.interpret_tapes(text_tapes(pset, GOLDEN_TEXTS), matrix, precision="float64")
    assert digest(explicit) == GOLDEN_DIGEST
    assert results.dtype == numpy.float64


def test_float32_storage_is_bit_identical_on_the_opcode_backend():
    pset = precision_pset()
    tapes = _all_tapes(pset)
    m32 = precision_matrix()
    expected = gp.interpret_tapes(tapes, m32.astype(numpy.float64))
    actual = gp.interpret_tapes(tapes, m32, precision="float32")
    assert actual.dtype == numpy.float64
    assert actual.shape == (len(tapes), m32.shape[0])
    assert numpy.array_equal(actual, expected, equal_nan=True)


def test_a_float64_matrix_is_rounded_to_float32_once():
    pset = precision_pset()
    tapes = text_tapes(pset, GOLDEN_TEXTS)
    m64 = precision_matrix().astype(numpy.float64) + 1e-9
    actual = gp.interpret_tapes(tapes, m64, precision="float32")
    expected = gp.interpret_tapes(tapes, m64.astype(numpy.float32).astype(numpy.float64))
    assert numpy.array_equal(actual, expected, equal_nan=True)


@pytest.mark.parametrize("precision", ["float16", "f4", "", None, numpy.float32])
def test_an_unknown_precision_is_rejected(precision):
    pset = precision_pset()
    tapes = text_tapes(pset, GOLDEN_TEXTS[:1])
    matrix = precision_matrix()
    with pytest.raises(ValueError, match="precision must be 'float64' or 'float32'"):
        gp.interpret_tapes(tapes, matrix, precision=precision)


def test_the_opcode_backend_widens_columns_without_a_matrix_copy():
    names = [f"c{index}" for index in range(64)]
    pset = gp.columnar_pset(names, window=None)
    tapes = text_tapes(pset, ["vadd(c0, c1)", "rolling_mean(vmul(c2, c0), 4)"])
    m32 = numpy.random.default_rng(5).normal(size=(100_000, 64)).astype(numpy.float32)
    gp.interpret_tapes(tapes, m32[:10], precision="float32")

    peak = _peak_bytes(lambda: gp.interpret_tapes(tapes, m32, precision="float32"))

    assert peak < m32.size * 8 // 2
    expected = gp.interpret_tapes(tapes, m32.astype(numpy.float64))
    actual = gp.interpret_tapes(tapes, m32, precision="float32")
    assert numpy.array_equal(actual, expected, equal_nan=True)


def test_as_matrix_packs_at_the_requested_precision():
    m32 = precision_matrix()
    assert as_matrix((m32,), 3, "float32") is m32
    assert as_matrix((m32,), 3).dtype == numpy.float64
    packed = as_matrix(tuple(m32.astype(numpy.float64).T), 3, "float32")
    assert packed.dtype == numpy.float32
    assert packed.flags["C_CONTIGUOUS"]
    assert numpy.array_equal(packed, m32, equal_nan=True)
    with pytest.raises(ValueError, match="precision"):
        as_matrix((m32,), 3, BAD_PRECISION)


def test_bounds_from_matrix_reads_a_float32_table_without_a_copy():
    m32 = precision_matrix()
    assert numpy.array_equal(bounds_from_matrix(m32), bounds_from_matrix(m32.astype(numpy.float64)))
    assert bounds_from_matrix(m32).dtype == numpy.float64
    large = numpy.ones((100_000, 8), dtype=numpy.float32)
    assert _peak_bytes(lambda: bounds_from_matrix(large)) < large.nbytes


def test_suffix_rescore_passes_a_float32_matrix_through(monkeypatch):
    pset = precision_pset()
    tapes = text_tapes(pset, GOLDEN_TEXTS)
    m32 = precision_matrix()
    prefix = gp.interpret_tapes(tapes, m32[:80], precision="float32")
    expected = gp.suffix_rescore(tapes, m32.astype(numpy.float64), prefix)
    seen = _spy(monkeypatch, suffix_rescore)

    actual = gp.suffix_rescore(tapes, m32, prefix, precision="float32")

    assert numpy.array_equal(actual, expected, equal_nan=True)
    assert seen[0].dtype == numpy.float32
    assert numpy.shares_memory(seen[0], m32)
    with pytest.raises(ValueError, match="precision"):
        gp.suffix_rescore(tapes, m32, prefix, precision=BAD_PRECISION)


def test_evaluate_columnar_scores_a_float32_matrix_without_a_copy(monkeypatch):
    pset = precision_pset()
    trees = [gp.PrimitiveTree.from_string(text, pset) for text in GOLDEN_TEXTS[:8]]
    m32 = precision_matrix()
    target = numpy.nan_to_num(m32[:, 0].astype(numpy.float64)) * 0.5
    expected = gp.evaluate_columnar(trees, pset, m32.astype(numpy.float64), target)
    seen = _spy(monkeypatch, columnar_setup)

    actual = gp.evaluate_columnar(trees, pset, m32, target, precision="float32")

    assert actual == expected
    assert all(numpy.shares_memory(matrix, m32) for matrix in seen)
    with pytest.raises(ValueError, match="precision"):
        gp.evaluate_columnar(trees, pset, m32, target, precision=BAD_PRECISION)


@pytest.fixture
def semantic_ind():
    creator.create_type(SEMANTIC_FIT, Fitness, weights=(-1.0,))
    creator.create_type(SEMANTIC_IND, gp.PrimitiveTree, fitness=creator.__dict__[SEMANTIC_FIT])
    yield creator.__dict__[SEMANTIC_IND]
    del creator.__dict__[SEMANTIC_FIT]
    del creator.__dict__[SEMANTIC_IND]


def test_semantic_crossover_scores_a_float32_matrix_without_a_copy(monkeypatch, semantic_ind):
    pset = precision_pset()
    first = semantic_ind(gp.PrimitiveTree.from_string("vadd(x, rolling_mean(y, 3))", pset))
    second = semantic_ind(gp.PrimitiveTree.from_string("vmul(s, vsub(y, x))", pset))
    m32 = precision_matrix()
    seen = _spy(monkeypatch, cx_semantic_one_point)
    tools.rng.seed(7)

    gp.cx_one_point_semantic(first, second, pset, m32, precision="float32")

    assert seen
    assert all(numpy.shares_memory(matrix, m32) for matrix in seen)
    with pytest.raises(ValueError, match="precision"):
        gp.cx_one_point_semantic(first, second, pset, m32, precision=BAD_PRECISION)


def test_the_interpret_tapes_policy_action_forwards_the_precision():
    m32 = precision_matrix()
    tapes = text_tapes(precision_pset(), GOLDEN_TEXTS[:2])
    result = tools.apply_policy_action(
        "interpret_tapes", tapes=tapes, matrix=m32, precision="float32"
    )
    assert result.applied
    with pytest.raises(ValueError, match="precision"):
        tools.apply_policy_action(
            "interpret_tapes", tapes=tapes, matrix=m32, precision=BAD_PRECISION
        )

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

import numpy
import pytest
from deap_er import gp, tools


def _typed_set():
    pset = gp.PrimitiveSetTyped("main", [float, float], float)
    pset.add_primitive(operator.add, [float, float], float, name="vadd")
    pset.add_primitive(operator.mul, [float, float], float, name="vmul")
    return pset


def _trees(pset):
    compact = gp.PrimitiveTree.from_string("vadd(ARG0, ARG1)", pset)
    bloated = gp.PrimitiveTree.from_string("vadd(vadd(ARG0, ARG1), vmul(ARG0, ARG1))", pset)
    return compact, bloated


def test_structural_meta_case_columns_size_and_depth():
    pset = _typed_set()
    compact, bloated = _trees(pset)
    matrix = tools.structural_meta_case_columns(
        [compact, bloated],
        prim_set=pset,
        columns=("size", "depth"),
    )
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(3.0)
    assert matrix[1, 0] == pytest.approx(7.0)
    assert matrix[0, 1] <= matrix[1, 1]


def test_structural_meta_case_columns_non_finite_fraction():
    pset = _typed_set()
    compact, _ = _trees(pset)
    predicted = numpy.array([[numpy.nan, 1.0, numpy.inf, 2.0]], dtype=numpy.float64)
    matrix = tools.structural_meta_case_columns(
        [compact],
        prim_set=pset,
        predicted=predicted,
        columns=("non_finite_fraction",),
    )
    assert matrix[0, 0] == pytest.approx(0.5)


def test_structural_meta_case_columns_missing_predicted_is_nan():
    pset = _typed_set()
    compact, _ = _trees(pset)
    matrix = tools.structural_meta_case_columns(
        [compact],
        prim_set=pset,
        columns=("non_finite_fraction",),
    )
    assert numpy.isnan(matrix[0, 0])


def test_structural_meta_case_columns_promote_hits():
    pset = _typed_set()
    tree = gp.PrimitiveTree.from_string("vadd(ARG0, vmul(ARG1, ARG0))", pset)
    gp.promote_subtree(pset, tree, index=2)
    using_promo = gp.PrimitiveTree.from_string("promo0(ARG0, ARG1)", pset)
    matrix = tools.structural_meta_case_columns(
        [using_promo],
        prim_set=pset,
        columns=("promote_hits",),
    )
    assert matrix[0, 0] == pytest.approx(1.0)


def test_structural_meta_case_columns_unique_opcodes_dedup(monkeypatch):
    pset = _typed_set()
    compact, _ = _trees(pset)
    duplicate_copy = gp.PrimitiveTree(compact)
    calls = {"count": 0}
    real_lower = gp.lower_tree

    def counted(tree, prim_set, *, fill=None):
        calls["count"] += 1
        return real_lower(tree, prim_set, fill=fill)

    monkeypatch.setattr(
        "deap_er.private.various.structural_meta_case.lower_tree",
        counted,
    )
    tools.structural_meta_case_columns(
        [compact, duplicate_copy],
        prim_set=pset,
        columns=("unique_opcodes",),
    )
    assert calls["count"] == 1


def test_structural_meta_case_weights_signs():
    weights = tools.structural_meta_case_weights()
    assert weights == (-1.0, -1.0, -1.0, -1.0, 1.0)
    assert tools.structural_meta_case_weights(("size", "non_finite_fraction")) == (-1.0, 1.0)


def test_structural_meta_case_columns_empty_population_raises():
    with pytest.raises(ValueError, match="non-empty"):
        tools.structural_meta_case_columns([])


def test_structural_meta_case_columns_unknown_column_raises():
    pset = _typed_set()
    compact, _ = _trees(pset)
    with pytest.raises(ValueError, match="unknown structural columns"):
        tools.structural_meta_case_columns([compact], prim_set=pset, columns=("bloat",))


def test_structural_meta_case_columns_requires_prim_set_for_opcode_columns():
    pset = _typed_set()
    compact, _ = _trees(pset)
    with pytest.raises(ValueError, match="prim_set is required"):
        tools.structural_meta_case_columns([compact], columns=("unique_opcodes",))

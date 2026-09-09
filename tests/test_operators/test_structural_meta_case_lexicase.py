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
from deap_er import Fitness, creator, gp, tools
from deap_er.private.operators.sel_lexicase_matrix import resolve_case_weights

FIT = "META_CASE_FIT"
IND = "META_CASE_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    yield creator.__dict__[IND]
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def _typed_set():
    pset = gp.PrimitiveSetTyped("main", [float, float], float)
    pset.add_primitive(operator.add, [float, float], float, name="vadd")
    pset.add_primitive(operator.mul, [float, float], float, name="vmul")
    return pset


def _population(ind_cls):
    pset = _typed_set()
    compact = gp.PrimitiveTree.from_string("vadd(ARG0, ARG1)", pset)
    bloated = gp.PrimitiveTree.from_string("vadd(vadd(ARG0, ARG1), vmul(ARG0, ARG1))", pset)
    compact_ind = ind_cls(compact)
    bloated_ind = ind_cls(bloated)
    compact_ind.fitness.values = (0.5,)
    bloated_ind.fitness.values = (0.5,)
    predicted = numpy.array(
        [
            [numpy.nan, numpy.nan, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=numpy.float64,
    )
    return pset, compact_ind, bloated_ind, predicted


def test_resolve_case_weights_requires_extension(ind_cls):
    pset, compact, bloated, _ = _population(ind_cls)
    meta = tools.structural_meta_case_columns(
        [compact, bloated],
        prim_set=pset,
        columns=("size",),
    )
    matrix = numpy.hstack([tools.fitness_case_matrix([compact, bloated]), meta])
    with pytest.raises(ValueError, match="pass fit_weights"):
        resolve_case_weights([compact, bloated], matrix, None)


def test_sel_lexicase_prefers_compact_tree_on_size_meta_case(ind_cls):
    pset, compact, bloated, _ = _population(ind_cls)
    meta = tools.structural_meta_case_columns(
        [compact, bloated],
        prim_set=pset,
        columns=("size",),
    )
    matrix = numpy.hstack([tools.fitness_case_matrix([compact, bloated]), meta])
    weights = (-1.0,) + tools.structural_meta_case_weights(("size",))
    chosen = tools.sel_lexicase(
        [compact, bloated],
        1,
        cases=[1],
        matrix=matrix,
        trust_matrix=True,
        fit_weights=weights,
    )
    assert chosen[0] is compact


def test_sel_lexicase_prefers_sparse_output_on_non_finite_meta_case(ind_cls):
    pset, compact, bloated, predicted = _population(ind_cls)
    matrix = numpy.hstack(
        [
            tools.fitness_case_matrix([compact, bloated]),
            tools.structural_meta_case_columns(
                [compact, bloated],
                prim_set=pset,
                predicted=predicted,
                columns=("non_finite_fraction",),
            ),
        ]
    )
    weights = (-1.0,) + tools.structural_meta_case_weights(("non_finite_fraction",))
    chosen = tools.sel_lexicase(
        [compact, bloated],
        1,
        cases=[1],
        matrix=matrix,
        trust_matrix=True,
        fit_weights=weights,
    )
    assert chosen[0] is compact

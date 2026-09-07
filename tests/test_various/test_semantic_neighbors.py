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
import math

import numpy
import pytest
from deap_er import Fitness, creator, gp, tools

FIT = "SEM_NN_FIT"
IND = "SEM_NN_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    yield creator.__dict__[IND]
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def test_semantic_distance_one_d_returns_length_one_array():
    result = tools.semantic_distance([1.0, 0.0], [1.0, 0.0], metric="euclidean")
    assert isinstance(result, numpy.ndarray)
    assert result.shape == (1,)
    assert result[0] == pytest.approx(0.0)


def test_semantic_distance_euclidean_and_cosine():
    query = numpy.array([1.0, 0.0, 0.0])
    matrix = numpy.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    euclid = tools.semantic_distance(query, matrix, metric="euclidean")
    cosine = tools.semantic_distance(query, matrix, metric="cosine")
    numpy.testing.assert_allclose(euclid, [0.0, math.sqrt(2.0)])
    numpy.testing.assert_allclose(cosine, [0.0, 1.0])
    assert tools.semantic_distance(query, matrix[0], metric="euclidean")[0] == pytest.approx(0.0)


def test_semantic_distance_ignores_warmup_columns():
    query = numpy.array([99.0, 1.0, 2.0])
    matrix = numpy.array([[-99.0, 1.0, 2.0], [0.0, 4.0, 6.0]])
    valid = numpy.array([False, True, True])
    dist = tools.semantic_distance(query, matrix, metric="euclidean", valid=valid)
    numpy.testing.assert_allclose(dist, [0.0, math.sqrt(3.0**2 + 4.0**2)])


def test_semantic_distance_empty_overlap_is_inf():
    query = numpy.array([math.nan, 1.0])
    matrix = numpy.array([[1.0, math.nan]])
    assert tools.semantic_distance(query, matrix, metric="euclidean")[0] == math.inf
    assert tools.semantic_distance(query, matrix, metric="cosine")[0] == math.inf


def test_semantic_nearest_ties_take_lowest_index():
    query = numpy.array([0.0, 0.0])
    matrix = numpy.array([[1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    nearest = gp.semantic_nearest(query, matrix, k=2)
    numpy.testing.assert_array_equal(nearest, [0, 1])


def test_semantic_nearest_skips_infinite_and_honors_k():
    query = numpy.array([math.nan, 1.0])
    matrix = numpy.array([[1.0, math.nan], [0.0, 1.0], [0.0, 3.0]])
    nearest = tools.semantic_nearest(query, matrix, k=2)
    numpy.testing.assert_array_equal(nearest, [1, 2])


def test_semantic_nearest_rejects_non_positive_k():
    with pytest.raises(ValueError, match="k must be at least 1"):
        tools.semantic_nearest([0.0], numpy.array([[0.0]]), k=0)


def test_semantic_nearest_trust_matrix(ind_cls):
    matrix = numpy.array([[0.0, 1.0], [1.0, 0.0]])
    people = [ind_cls([]), ind_cls([])]
    people[0].fitness.values = (1.0,)
    with pytest.raises(ValueError, match="valid fitness"):
        tools.semantic_nearest([0.0, 1.0], matrix, individuals=people, trust_matrix=False)
    nearest = tools.semantic_nearest([0.0, 1.0], matrix, individuals=people, trust_matrix=True)
    numpy.testing.assert_array_equal(nearest, [0])
    assert isinstance(nearest, numpy.ndarray)
    assert numpy.issubdtype(nearest.dtype, numpy.integer)
    assert not isinstance(nearest[0], type(people[0]))
    assert people[int(nearest[0])] is people[0]

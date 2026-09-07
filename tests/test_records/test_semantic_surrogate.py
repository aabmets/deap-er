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
from typing import Any, cast

import numpy
import pytest
from deap_er import Fitness, creator, tools

FIT = "SEM_SUR_FIT"
IND = "SEM_SUR_IND"


@pytest.fixture
def ind_cls():
    creator.create_type(FIT, Fitness, weights=(-1.0,))
    creator.create_type(IND, list, fitness=creator.__dict__[FIT])
    yield creator.__dict__[IND]
    del creator.__dict__[FIT]
    del creator.__dict__[IND]


def test_surrogate_nearest_and_linear_predict():
    store = tools.SemanticSurrogate()
    matrix = numpy.array([[1.0], [2.0], [4.0]])
    values = numpy.array([2.0, 4.0, 8.0])
    store.update(matrix, values)
    assert store.nearest([2.0])[0] == 1
    assert store.predict([2.0], kind="nearest") == pytest.approx(4.0)
    assert store.predict([2.0], kind="nearest", k=2) == pytest.approx(3.0)
    assert store.predict([3.0], kind="linear") == pytest.approx(6.0)


def test_surrogate_linear_underdetermined_uses_min_norm():
    store = tools.SemanticSurrogate()
    store.update(numpy.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), [1.0, 1.0])
    query = [1.0, 1.0, 0.0]
    assert store.predict(query, kind="nearest") == pytest.approx(1.0)
    assert store.predict(query, kind="linear") == pytest.approx(2.0)


def test_surrogate_linear_falls_back_when_rank_zero():
    store = tools.SemanticSurrogate()
    store.update(numpy.array([[1.0, 2.0], [3.0, 4.0]]), [1.0, 2.0])
    valid = numpy.array([False, False])
    assert math.isnan(store.predict([1.0, 2.0], kind="linear", valid=valid))


def test_surrogate_update_does_not_write_fitness(ind_cls):
    store = tools.SemanticSurrogate()
    first = ind_cls([1])
    second = ind_cls([2])
    first.fitness.values = (1.5,)
    second.fitness.values = (2.5,)
    before = (first.fitness.values, second.fitness.values)
    store.update(
        numpy.array([[0.0], [1.0]]),
        [1.5, 2.5],
        individuals=[first, second],
    )
    assert first.fitness.values == before[0]
    assert second.fitness.values == before[1]
    assert store.predict([1.0], kind="nearest") == pytest.approx(2.5)


def test_surrogate_requires_update_and_known_kind():
    store = tools.SemanticSurrogate()
    with pytest.raises(ValueError, match="update must be called"):
        store.predict([0.0])
    store.update(numpy.array([[0.0]]), [1.0])
    unknown_kind = cast(Any, "ridge")
    with pytest.raises(ValueError, match="nearest"):
        store.predict([0.0], kind=unknown_kind)


def test_surrogate_trust_matrix(ind_cls):
    store = tools.SemanticSurrogate()
    bare = ind_cls([])
    with pytest.raises(ValueError, match="valid fitness"):
        store.update(numpy.array([[0.0]]), [1.0], individuals=[bare], trust_matrix=False)
    store.update(numpy.array([[0.0]]), [1.0], individuals=[bare], trust_matrix=True)
    assert store.predict([0.0], kind="nearest") == pytest.approx(1.0)

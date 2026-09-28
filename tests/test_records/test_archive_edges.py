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
import pytest
from deap_er import Fitness, creator, tools

MO_FIT = "EDGE_MO_FIT"
MO_IND = "EDGE_MO_IND"
SO_FIT = "EDGE_SO_FIT"
SO_IND = "EDGE_SO_IND"


@pytest.fixture
def mo_cls():
    creator.create_type(MO_FIT, Fitness, weights=(1.0, 1.0))
    creator.create_type(MO_IND, list, fitness=creator.__dict__[MO_FIT])
    yield creator.__dict__[MO_IND]
    del creator.__dict__[MO_FIT]
    del creator.__dict__[MO_IND]


@pytest.fixture
def so_cls():
    creator.create_type(SO_FIT, Fitness, weights=(1.0,))
    creator.create_type(SO_IND, list, fitness=creator.__dict__[SO_FIT])
    yield creator.__dict__[SO_IND]
    del creator.__dict__[SO_FIT]
    del creator.__dict__[SO_IND]


def _archives():
    return [
        tools.GridArchive([(0.0, 1.0)], 2),
        tools.CvtArchive([[0.0], [1.0]]),
        tools.UnstructuredArchive(1, 0.5),
    ]


@pytest.mark.parametrize("archive", _archives(), ids=["grid", "cvt", "unstructured"])
def test_multi_objective_rejected_even_without_valid_fitness(mo_cls, archive):
    unevaluated = mo_cls([1])
    with pytest.raises(ValueError, match="single-objective"):
        archive.add(unevaluated, [0.5])
    with pytest.raises(ValueError, match="single-objective"):
        archive.add(unevaluated, [float("nan")])


def test_grid_rejects_range_with_overflowing_span():
    with pytest.raises(ValueError, match="span"):
        tools.GridArchive([(-1e308, 1e308)], 10)


def test_grid_add_at_upper_bound_of_huge_finite_range(so_cls):
    archive = tools.GridArchive([(-1e307, 1e307)], 10)
    ind = so_cls([1])
    ind.fitness.values = (1.0,)
    assert archive.add(ind, [1e307])
    assert (9,) in archive


@pytest.mark.parametrize("n_centroids", [2, 600])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_cvt_nearest_centroid_rejects_non_finite(n_centroids, value):
    archive = tools.CvtArchive([[float(i)] for i in range(n_centroids)])
    with pytest.raises(ValueError, match="finite"):
        archive.nearest_centroid([value])
